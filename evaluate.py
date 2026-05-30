import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://127.0.0.1:8000/chat"

# test cases with expected keywords that must appear in a good answer
TEST_CASES = [
    {
        "question": "My worker is showing in Fieldglass but not in Workday, what should I do?",
        "expected_keywords": ["provision", "workday hr", "servicenow", "ticket"],
        "expected_source": "error_runbook.pdf"
    },
    {
        "question": "The manager tagged in Fieldglass is incorrect, how do I fix it?",
        "expected_keywords": ["manager", "workday", "requisition", "update"],
        "expected_source": "error_runbook.pdf"
    },
    {
        "question": "Worker start date is wrong in Workday, what went wrong?",
        "expected_keywords": ["date", "fieldglass", "timezone", "re-trigger"],
        "expected_source": "error_runbook.pdf"
    },
    {
        "question": "Who should I contact for a cost center not found error?",
        "expected_keywords": ["finance", "cost center", "email"],
        "expected_source": "escalation_guide.pdf"
    },
    {
        "question": "What happens when I close an engagement in Fieldglass?",
        "expected_keywords": ["terminated", "workday", "termination"],
        "expected_source": "field_mapping.pdf"
    },
    {
        "question": "What is the weather like today?",
        "expected_keywords": ["don't have", "servicenow", "information"],
        "expected_source": None  # LEO should say it doesn't know
    },
]

def evaluate():
    print("=== LEO — Evaluation Report ===\n")
    print("Make sure uvicorn is running before this script!\n")

    results = []
    passed = 0

    for i, test in enumerate(TEST_CASES):
        print(f"Test {i+1}/{len(TEST_CASES)}: {test['question'][:55]}...")

        try:
            response = requests.post(
                API_URL,
                json={"question": test["question"], "chat_history": []},
                timeout=30
            )
            data = response.json()
            answer = data["answer"].lower()
            sources = [s.lower() for s in data.get("sources", [])]

            # check keyword relevancy
            keywords_found = [k for k in test["expected_keywords"] if k.lower() in answer]
            relevancy_score = len(keywords_found) / len(test["expected_keywords"])

            # check source accuracy
            if test["expected_source"]:
                source_correct = any(test["expected_source"] in s for s in sources)
            else:
                source_correct = True  # out of scope question, source doesn't matter

            # overall pass: relevancy > 50% and correct source
            test_passed = relevancy_score >= 0.5 and source_correct
            if test_passed:
                passed += 1

            result = {
                "question": test["question"],
                "relevancy_score": round(relevancy_score, 2),
                "keywords_found": keywords_found,
                "keywords_missing": [k for k in test["expected_keywords"] if k not in keywords_found],
                "source_correct": source_correct,
                "passed": test_passed,
                "answer_preview": data["answer"][:120] + "..."
            }
            results.append(result)

            status = "PASS" if test_passed else "FAIL"
            print(f"  Status:    {status}")
            print(f"  Relevancy: {relevancy_score:.0%} ({len(keywords_found)}/{len(test['expected_keywords'])} keywords found)")
            print(f"  Source:    {'correct' if source_correct else 'wrong'}")
            print()

        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({"question": test["question"], "error": str(e), "passed": False})

    # summary
    total = len(TEST_CASES)
    score = passed / total
    print("=" * 40)
    print(f"OVERALL SCORE: {passed}/{total} tests passed ({score:.0%})")
    if score >= 0.8:
        print("Rating: GOOD — LEO is answering reliably")
    elif score >= 0.6:
        print("Rating: OKAY — Some answers need improvement")
    else:
        print("Rating: NEEDS WORK — Check retrieval and prompt")
    print("=" * 40)

    # save
    output = {
        "score": round(score, 2),
        "passed": passed,
        "total": total,
        "results": results
    }
    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nFull results saved to eval_results.json")

if __name__ == "__main__":
    evaluate()
