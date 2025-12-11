import re
from typing import List

from mathruler.grader import extract_boxed_content, grade_answer
from swift.plugin import ORM, orms


class PAPOFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
        matches = [re.fullmatch(pattern, response) for response in completions]
        return [1.0 if match else 0.0 for match in matches]


class PAPOAccuracy(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            answer = extract_boxed_content(content)
            rewards.append(1.0 if grade_answer(answer, sol) else 0.0)
        return rewards


orms['papo_format'] = PAPOFormat
orms['papo_accuracy'] = PAPOAccuracy
