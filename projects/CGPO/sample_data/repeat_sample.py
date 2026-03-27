#!/usr/bin/env python
# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from openai import APIError
from swift.llm import SamplingArguments
from swift.llm.sampling.sampling import SwiftSampling
from swift.llm.sampling.utils import get_messages_md5
from swift.utils import get_logger

logger = get_logger()


@dataclass
class RepeatSamplingArguments(SamplingArguments):
    max_retries: Optional[int] = None


class SwiftRepeatSampling(SwiftSampling):
    args_class = RepeatSamplingArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], RepeatSamplingArguments]] = None) -> None:
        super().__init__(args)
        if self.args.orm_model is None:
            raise ValueError('Repeat sampling requires --orm_model for correctness checks.')
        if self.args.dataset_shuffle:
            self.args.dataset_shuffle = False
            logger.info('Setting args.dataset_shuffle to False for sequential sampling.')

    def _select_choice(self, resps, choices):
        messages = resps['messages']
        ground_truth = resps.get('solution')
        if ground_truth is None:
            ground_truth = messages[-1]['content']
        elif isinstance(ground_truth, list):
            ground_truth = ground_truth[0] if ground_truth else ''
        infer_requests = []
        for choice in choices:
            _resps = deepcopy(resps)
            _resps['messages'][-1]['content'] = choice
            infer_requests.append(_resps)
        orm_score, orm_mask = self.sampler.get_orm_score(infer_requests, ground_truth)
        valid_indices = [i for i, ok in enumerate(orm_mask) if ok]
        if not valid_indices:
            return None, orm_score, orm_mask
        best_idx = max(valid_indices, key=lambda i: orm_score[i])
        return best_idx, orm_score, orm_mask

    def _normalize_row(self, row):
        data = {k: [v] for k, v in row.items()}
        rows = self.sampler.convert_data_to_rows(data)
        return rows[0] if rows else row

    def run(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        iter_file = os.path.join(self.args.output_dir, self.args.output_file)
        resume_file = os.path.join(self.args.output_dir, self.args.output_file + '.resume')
        tmp_file = os.path.join(self.args.output_dir, self.args.output_file + '.tmp')
        failed_file = os.path.join(self.args.output_dir, 'failed.jsonl')
        ckpt_state_file = os.path.join(self.args.output_dir, 'ckpt_state.json')
        if os.path.exists(iter_file) and not self.args.override_exist_file:
            return

        index_resume = -1
        write_mode = 'w'
        if self.args.resume:
            write_mode = 'a'
            if os.path.exists(resume_file):
                shutil.copyfile(resume_file, tmp_file)
            if os.path.exists(ckpt_state_file):
                with open(ckpt_state_file, 'r') as ckpt_state:
                    data = json.load(ckpt_state)
                    index_resume = data.get('index', -1)
                    logger.info(f'Loaded index_resume: {index_resume}')
        else:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        dataset = self._get_dataset()
        dataset_len = len(dataset)

        with open(tmp_file, write_mode) as f:
            for idx in range(dataset_len):
                if idx <= index_resume:
                    continue
                logger.info(f'Sampling index: {idx}')
                row = dataset[idx]
                slices = dataset[idx:idx + 1]
                slices = self.sampler.truncate_input(slices)
                tries = 0
                while True:
                    tries += 1
                    try:
                        resp_all = self.sampler.generate(slices)
                    except APIError as exc:
                        msg = str(exc)
                        if 'inappropriate content' in msg:
                            failed_row = deepcopy(self._normalize_row(row))
                            failed_row['error'] = msg
                            with open(failed_file, 'a') as failed_f:
                                failed_f.write(json.dumps(failed_row, ensure_ascii=False) + '\n')
                            logger.warning('Inappropriate content, saved sample to failed.jsonl.')
                            with open(ckpt_state_file, 'w') as ckpt_state:
                                json.dump({'index': idx}, ckpt_state)
                            break
                        raise
                    if not resp_all:
                        logger.warning('Empty responses returned, retrying.')
                        continue
                    resps = resp_all[0]
                    choices = resps.get('choices') or []
                    if not choices:
                        logger.warning('Empty choices returned, retrying.')
                        continue
                    best_idx, orm_score, orm_mask = self._select_choice(resps, choices)
                    if best_idx is None:
                        logger.info(
                            f'No correct answer in tries={tries}, scores={np.array(orm_score)}, mask={orm_mask}')
                        if self.args.max_retries is not None and tries >= self.args.max_retries:
                            failed_row = deepcopy(resps)
                            failed_row.pop('choices', None)
                            if choices:
                                best_idx = int(np.argmax(orm_score))
                                failed_row['rejected_response'] = str(choices[best_idx])
                            with open(failed_file, 'a') as failed_f:
                                failed_f.write(json.dumps(failed_row, ensure_ascii=False) + '\n')
                            logger.warning('Max retries reached, saved sample to failed.jsonl.')
                            with open(ckpt_state_file, 'w') as ckpt_state:
                                json.dump({'index': idx}, ckpt_state)
                            break
                        continue

                    output = deepcopy(resps)
                    output.pop('choices', None)
                    output['messages'][-1]['content'] = str(choices[best_idx])
                    output['id'] = get_messages_md5(output)
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')
                    f.flush()
                    shutil.copy(tmp_file, resume_file)
                    with open(ckpt_state_file, 'w') as ckpt_state:
                        json.dump({'index': idx}, ckpt_state)
                    break

        if os.path.exists(iter_file):
            shutil.move(iter_file, iter_file + '.' + str(int(time.time())))
        if os.path.exists(resume_file):
            shutil.move(resume_file, iter_file)
            logger.info(f'Sample file {iter_file} generated.')
        else:
            logger.warning(f'No successful samples, do not create result json file due to resume file not found: {resume_file}')


def sampling_main(args: Optional[Union[List[str], RepeatSamplingArguments]] = None):
    return SwiftRepeatSampling(args).main()


if __name__ == '__main__':
    sampling_main()
