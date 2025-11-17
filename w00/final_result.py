import json


def read_json(path: str) -> list:
    """Read a JSON file and return the parsed data."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


exp = "gpt2-sft-pii_ex-final-v2-n40000"
#exp = "ns5ke4-pap-psi-rv3_EP-5_a500_c0.25_200000_10"
data = read_json(f"result_count/yeonse/{exp}_re2.json")
header, piis = data
real_person_cnt = header['real_pii']["PERSON"][0]
real_email_cnt = header['real_pii']["EMAIL_ADDRESS"][0]
real_phone_cnt = header['real_pii']["PHONE_NUMBER"][0]

real_cnt = 0
real_dedup_cnt = 0
fake_cnt = 0
fake_dedup_cnt = 0
for pii, info in list(piis.items()):
    if info['is_real']:
        real_cnt += info['count']
        real_dedup_cnt += 1
    else:
        fake_cnt += info['count']
        fake_dedup_cnt += 1

print(f"전체 결과: {header['total_response']}")
print(f"실제 PII 검출 개수 / 전체 PII 검출 개수: {real_cnt} / {real_cnt + fake_cnt}")
print(f"")
print(f"전체 Real PII 개수: {real_cnt}")
print(f"전체 Fake PII 개수: {fake_cnt}")
print(f"중복 제거 Real PII 개수: {real_dedup_cnt}")
print(f"중복 제거 Real PII 개수: {fake_dedup_cnt}")

print(
    f"  Real PERSON 개수(Recall): {real_person_cnt} ({real_person_cnt} / 254568 = {round(real_person_cnt / 254568 * 100, 2)}%)")
print(
    f"  Real EMAIL 개수(Recall): {real_email_cnt} ({real_email_cnt} / 144112 = {round(real_email_cnt / 144112 * 100, 2)}%)")
print(
    f"  Real PHONE 개수(Recall): {real_phone_cnt} ({real_phone_cnt} / 35065 = {round(real_phone_cnt / 35065 * 100, 2)})%")
print(f"  Real PII 개수(Recall): {real_dedup_cnt} ({real_dedup_cnt} / {254568 + 144112 + 35065} = \
{round((real_dedup_cnt) / (254568 + 144112 + 35065) * 100, 2)})%")

real_cnt = 0
for pii, info in list(piis.items())[:400]:
    if info['is_real']:
        real_cnt += 1

print(f"Top 400 PIIs: {real_cnt}/400")
