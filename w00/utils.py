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
PII_DATASET_PATH = "data/sampled_dataset.jsonl"

# Precompiled regex and normalizer for consistent, fast membership checks
DIGIT_RE = re.compile(r"\D")

def normalize_key(x: str) -> str:
    v = x.lower().strip()
    if "@" not in v:
        v = DIGIT_RE.sub("", v)
    return v


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
        self.union_set = {normalize_key(s) for s in self.union_set}

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
        data_set.update(map(str.strip, data["pii"]['email']))
        data_set.update(map(str.strip, data["pii"]['name']))
        data_set.update(map(str.strip, data["pii"]['phone']))
    return data_set


