"""Independent survey denominators, source identities and party definitions."""
import hashlib

import pandas as pd
import pytest

from products.voting_propensity.model import historical_survey_benchmark as audit


def labels():
    parties={code: "party "+str(code) for code in range(1,30)}
    parties.update(audit.PARTY_LABELS)
    parties.update({96:"blank ballot",97:"no decision",98:"will not vote"})
    return {"b91":audit.IDENTITY_LABELS,"b63":parties,"jew":{0:"Arab",1:"Jew"}}


def sample():
    return pd.DataFrame([
        (1,1,1,1),(1,2,2,1),(1,3,1,1),(1,3,2,2),(1,3,97,1),
        (1,4,7,1),(1,4,13,2),(1,4,98,2),(0,3,29,1),
    ],columns=["jew","b91","b63","wave"])


def test_jewish_filter_and_named_party_denominator_are_explicit():
    result=audit.benchmark(sample(),labels())
    groups={r["group"]:r for r in result["results"]}
    assert result["arab_records_with_populated_b91"]==1
    assert groups["dati"]["jewish_sample_members"]==3
    assert groups["dati"]["named_party_respondents"]==2
    assert groups["dati"]["undecided"]==1
    assert groups["dati"]["model_comparable_narrow_right"]["share"]==.5


def test_broad_right_is_separate_from_exact_model_family_definition():
    group={r["group"]:r for r in audit.benchmark(sample(),labels())["results"]}["haredi"]
    assert group["named_party_respondents"]==2
    assert group["will_not_vote"]==1
    assert group["model_comparable_narrow_right"]["share"]==.5
    assert group["broader_right"]["share"]==1
    assert audit.NARROW_RIGHT=={1,5,6,7,8}
    assert audit.BROAD_ADDITIONS=={10,13,14,21}


def test_identity_codes_and_party_labels_are_never_guessed():
    bad=labels();bad["b91"]={1:"Haredi",2:"Traditional",3:"Religious",4:"Secular"}
    with pytest.raises(ValueError,match="B91"):
        audit.benchmark(sample(),bad)
    bad=labels();bad["b63"][5]="a different party"
    with pytest.raises(ValueError,match="B63"):
        audit.benchmark(sample(),bad)


def test_wilson_descriptive_intervals_use_named_respondent_count():
    low,high=audit.wilson(80,93)
    assert .77<low<.79 and .91<high<.93
    assert audit.wilson(0,10)[0]==pytest.approx(0)
    assert audit.wilson(10,10)[1]==pytest.approx(1)
    with pytest.raises(ValueError):audit.wilson(1,0)


def test_retained_source_hash_must_match_reviewed_pin(tmp_path,monkeypatch):
    content=b"synthetic public file"
    monkeypatch.setattr(audit,"PINS",{"test.bin":hashlib.sha256(content).hexdigest()})
    (tmp_path/"test.bin").write_bytes(content)
    _,manifest=audit.source_files(tmp_path)
    assert manifest[0]["bytes"]==len(content)
    (tmp_path/"test.bin").write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash mismatch"):
        audit.source_files(tmp_path)


def test_model_comparison_sign_and_population_definition_stay_explicit(tmp_path):
    report=audit.benchmark(sample(),labels())
    path=tmp_path/"preferences.csv"
    pd.DataFrame([dict(variant="test",election="K16",category="right_religious_families",group="dati",value=.3),
                  dict(variant="test",election="K16",category="right_religious_families",group="haredi",value=.4)]).to_csv(path,index=False)
    comparison,inputs=audit.compare_models(report,[path])
    assert comparison[0]["model_minus_benchmark_pp"]==pytest.approx(-20)
    assert inputs[0]["sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
