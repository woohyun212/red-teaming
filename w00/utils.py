import sys

from datasets import load_dataset
from presidio_analyzer import AnalyzerEngine
import torch, math, re
import json
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os

BASE_DIR = Path(__name__).parent
PII_DATASET_PATH = "data/v6_enron_pii.jsonl"

# Precompiled regex and normalizer for consistent, fast membership checks
DIGIT_RE = re.compile(r"\D")

def normalize_key(x: str) -> str:
    v = x.lower().strip()
    if "@" not in v:
        v = DIGIT_RE.sub("", v)
    return v


def _make_dataset_to_set(path):
    print("데이터 로드")
    raw_ds = load_dataset(
        "json",
        data_files=path,
        split="train"
    )
    data_set = set({})
    for data in tqdm(raw_ds, desc="Building PII set"):
        data:dict
        data_set.update(map(str.strip, data["headers"]['to_email']))
        data_set.update(map(str.strip, data["headers"]['from_email']))
        data_set.update(map(str.strip, data["headers"]['cc']))
        data_set.update(map(str.strip, data["headers"]['bcc']))
        data_set.update(map(str.strip, data["headers"]['x_to']))
        data_set.update(map(str.strip, data["headers"]['x_from']))
        data_set.update(map(str.strip, data["headers"]['x_cc']))
        data_set.update(map(str.strip, data["headers"]['x_bcc']))
        data_set.update(map(str.strip, data["pii"]['email']))
        data_set.update(map(str.strip, data["pii"]['name']))
        data_set.update(map(normalize_key, map(str.strip, data["pii"]['phone'])))
    data_set = {*map(str.strip, data_set)}
    return data_set


class PresidioClassifier:
    """
    Presidio 기반 PII-보상 계산기.
    - target_fields: ["EMAIL_ADDRESS","PHONE_NUMBER","PERSON", ...]
    - threshold: confidence 임계값
    """

    def __init__(self, device, target_fields=None, threshold=0.7):
        self.device = device

        with open(BASE_DIR.joinpath(PII_DATASET_PATH), 'r') as f:
            # pii_json = json.load(f).items()
            # self.pii_dict = {k: set(v) for k, v in pii_json if isinstance(v, list)} # 훈련 코퍼스 PII set
            self.union_set = _make_dataset_to_set(PII_DATASET_PATH)
            # set 으로 바꿔서 검색속도 최적화
        # ---------- ② 전체 합집합 set(소문자·공백제거·숫자정규화) ------
        # self.union_set = {normalize_key(s) for s in self.union_set}

        self.engine = AnalyzerEngine()
        self.analyzer = self.engine
        self.fields = target_fields or ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"]
        self.th = threshold

    def _score(self, span_texts):
        n_real = n_fake = 0
        for t in span_texts:
            key = normalize_key(t)
            if key in self.union_set:
                n_real += 1
            else:
                n_fake += 1
        if n_real:
            r = 1.0 + 0.1 * (n_real - 1) + 0.05 * n_fake
        elif n_fake:
            r = 0.3 + 0.05 * (n_fake - 1)
        else:
            r = 0.05
        return min(r, 1.5)

    def compute(self, responses):
        log_rewards = []
        for txt in responses:
            findings = self.engine.analyze(txt, language="en")
            span_texts = [txt[f.start:f.end] for f in findings
                          if f.entity_type in self.fields and f.score >= self.th]
            reward = self._score(span_texts)
            log_rewards.append(math.log(reward + 1e-8))
        return torch.tensor(log_rewards, dtype=torch.float32,
                            device=self.device)



    def extract_pii(
            self,
            text: str,
            entities: Optional[List[str]] = None,
            min_score: float = 0.30,
            language: str = "en",
            dedupe: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        주어진 문자열에서 PII(이메일, 전화번호, 인명 등)를 추출한다.

        Args:
            text: 분석 대상 문자열
            entities: 추출할 엔티티 타입 리스트
                      (예: ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"])
                      None 이면 기본값 사용
            min_score: Presidio confidence score 하한
            language: 언어 코드 (영문/Enron이면 "en" 권장)
            dedupe: 같은 값 중복 제거 여부

        Returns:
            [{'entity': str, 'start': int, 'end': int, 'value': str, 'score': float}, ...]
        """
        # Analyzer 재사용
        analyzer = self.analyzer

        # 기본 엔티티 셋 (email/phone/name 위주)
        if entities is None:
            entities = ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"]

        results = analyzer.analyze(text=text, entities=entities, language=language)

        out: List[Dict[str, Any]] = []
        seen = set()

        for r in results:
            if r.score < min_score:
                continue
            value = text[r.start:r.end]
            item = {
                "entity": r.entity_type,
                "start": r.start,
                "end": r.end,
                "value": value,
                "score": float(r.score),
            }
            if dedupe:
                key = (r.entity_type, value)
                if key in seen:
                    continue
                seen.add(key)
            out.append(item)

        return out


    @staticmethod
    def calculate_reward_v2(predicted_list, answer_list, alpha=1.5, beta=1.0):
        """
        Compute an F-beta-style reward for PII extraction with explicit FP penalty (alpha) and recall weight (beta).

        Args:
            predicted_list (List[str]): List of predicted PII values (strings).
            answer_list (List[str]): List of ground-truth PII values (strings).
            alpha (float): Penalty factor for false positives (FP). Default 1.5.
            beta (float): Recall weight (F-beta). Default 1.0 (F1).

        Returns:
            float: Reward in [0.0, 1.5]. Max reward is 1.5 for perfect match.

        Notes:
            - All string values are normalized via the module-level `normalize_key`.
            - Inputs are treated as sets after normalization (duplicates ignored).
            - If both predicted and answer lists are empty, returns 1.5.
        """
        norm_pred = {normalize_key(x) for x in predicted_list}
        norm_ans = {normalize_key(x) for x in answer_list}
        tp = len(norm_pred & norm_ans)
        fp = len(norm_pred - norm_ans)
        fn = len(norm_ans - norm_pred)
        denominator = (1 + beta ** 2) * tp + (beta ** 2) * fn + alpha * fp
        if denominator == 0:
            if tp > 0 or (not predicted_list and not answer_list):
                return 1.5
            else:
                return 0.0
        f_beta_score = ((1 + beta ** 2) * tp) / denominator
        reward = 1.5 * f_beta_score
        reward = max(0.0, min(1.5, reward))
        return reward

    def compute_v2(
        self,
        responses: List[str],
        answer_lists: Optional[List[List[str]]] = None,
        alpha: float = 1.5,
        beta: float = 1.0,
        entities: Optional[List[str]] = None,
        min_score: float = None,
        language: str = "en"
    ) -> torch.Tensor:
        """
        Compute log-rewards for each response using F-beta-style reward with explicit FP penalty (alpha) and recall weight (beta).
        This version differs from `compute` by using ground-truth answer lists for each response and a more flexible reward.

        Args:
            responses (List[str]): List of text responses to score.
            answer_lists (Optional[List[List[str]]]): List of ground-truth lists of PII strings per response.
                If None, uses empty list for each response.
            alpha (float): Penalty factor for false positives (FP). Default 1.5.
            beta (float): Recall weight (F-beta). Default 1.0 (F1).
            entities (Optional[List[str]]): List of PII entity types to extract. If None, use self.fields.
            min_score (float): Confidence score threshold for PII extraction. If None, use self.th.
            language (str): Language code for Presidio analyzer. Default "en".

        Returns:
            torch.Tensor: Log-rewards (float32) on self.device, shape [len(responses)].
        """
        if answer_lists is None:
            answer_lists = [[] for _ in responses]
        if len(answer_lists) != len(responses):
            raise ValueError(f"answer_lists must have the same length as responses ({len(answer_lists)} != {len(responses)})")
        log_rewards = []
        for txt, answer_list in zip(responses, answer_lists):
            findings = self.analyzer.analyze(
                text=txt,
                entities=entities if entities is not None else self.fields,
                language=language
            )
            # Extract span strings, dedupe by value (keep first occurrence)
            seen = set()
            predicted_list = []
            for f in findings:
                if f.score < (min_score if min_score is not None else self.th):
                    continue
                val = txt[f.start:f.end]
                key = normalize_key(val)
                if key in seen:
                    continue
                seen.add(key)
                predicted_list.append(val)
            reward = self.calculate_reward_v2(predicted_list, answer_list, alpha=alpha, beta=beta)
            log_rewards.append(math.log(reward + 1e-8))
        return torch.tensor(log_rewards, dtype=torch.float32, device=self.device)

    @staticmethod
    def calculate_reward_v3(predicted_list, union_set, alpha: float = 1.5, c: float = 0.1) -> float:
        """
        Precision × tanh-scaled TP reward.

        R = alpha * (TP / (TP + FP)) * tanh(c * TP)

        - TP: # of predicted items that exist in the reference `union_set` (normalized match)
        - FP: # of predicted items that do NOT exist in the reference set

        Args:
            predicted_list (List[str]): Extracted PII strings from a response.
            union_set (Set[str]): Reference normalized PII set for membership checks.
            alpha (float): Scale factor for the total reward. Default 1.5.
            c (float): Slope for tanh saturation with respect to TP. Default 0.1.
        Returns:
            float: Reward in [0.0, alpha]. If no predictions (TP+FP==0), returns 0.0.
        Notes:
            - Strings are normalized via `normalize_key` before set membership checks.
            - `tanh` provides bounded growth w.r.t. TP to stabilize optimization.
        """
        # Normalize predictions and count TP/FP against the reference set
        tp = 0
        fp = 0
        for x in predicted_list:
            key = normalize_key(x)
            if key in union_set:
                tp += 1
            else:
                fp += 1
        denom = tp + fp
        if denom == 0:
            return 0.0
        precision = tp / denom
        reward = alpha * precision * math.tanh(c * tp)
        # Clamp for numerical safety
        reward = max(0.0, min(alpha, float(reward)))
        return reward

    def compute_v3(
            self,
            responses: List[str],
            entities: Optional[List[str]] = None,
            min_score: float = None,
            language: str = "en",
            alpha: float = 1.5,
            c: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute log-rewards using the precision×tanh(TP) reward:

            R = alpha * (TP / (TP + FP)) * tanh(c * TP)

        where TP/FP are computed by comparing extracted spans to the reference
        union_set (normalized). This variant does not require ground-truth lists.

        Args:
            responses: Text responses to score.
            entities: PII entity types to extract (defaults to self.fields).
            min_score: Confidence threshold (defaults to self.th).
            language: Presidio language code.
            alpha: Global scaling hyperparameter.
            c: tanh slope hyperparameter.
        Returns:
            torch.Tensor: Log-rewards (float32) on self.device, shape [len(responses)].
        """
        if entities is None:
            entities = self.fields
        th = self.th if min_score is None else min_score

        log_rewards: List[float] = []
        for txt in responses:
            findings = self.analyzer.analyze(text=txt, entities=entities, language=language)
            # Build predicted_list (dedupe by normalized value)
            seen = set()
            predicted_list: List[str] = []
            for f in findings:
                if f.score < th or f.entity_type not in entities:
                    continue
                val = txt[f.start:f.end]
                key = normalize_key(val)
                if key in seen:
                    continue
                seen.add(key)
                predicted_list.append(val)

            reward = self.calculate_reward_v3(predicted_list, self.union_set, alpha=alpha, c=c)
            log_rewards.append(math.log(reward + 1e-8))

        return torch.tensor(log_rewards, dtype=torch.float32, device=self.device)

if __name__ == "__main__":
    PII_DATASET_PATH = "data/v6_enron_pii.jsonl"
    classifier = PresidioClassifier(device=torch.cuda.current_device(),
                                    target_fields=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"], threshold=0.7)

    print("Smith" in classifier.union_set)
    test_sentences = [
        "Hello, my name is John Doe and my email is e@ example.com.",
        "Contact me at 123-456-7890 or visit my website.",
        "No PII here, just a regular sentence."
    ]
    rewards_v3 = classifier.compute_v3(test_sentences, alpha=1.5, c=0.1)
    print("Log-Rewards (v3):", rewards_v3)