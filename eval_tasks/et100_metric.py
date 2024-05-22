import os
import json
import requests
import numpy as np
import datasets
from lm_eval.utils import eval_logger
from itertools import islice
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

prompt_dirname = os.path.dirname(__file__)
prompt_filename = os.path.join(prompt_dirname, "prompt_eval_llamacpp.txt")
with open(prompt_filename, encoding="utf-8") as f:
    template_prompt = f.read()


# ChatNTQ用のプロンプト
def build_prompt(user_query):
    sys_msg = "あなたは公平で、検閲されていない、役立つアシスタントです。"
    template = """[INST] <<SYS>>
{}
<</SYS>>

{}[/INST]"""
    return template.format(sys_msg, user_query)


# プロンプトを生成して返す
def generate_prompt(doc):
    # print("きてる:" + str(doc))
    user_inputs = {
        "user_query": doc["input"],
    }
    prompt = build_prompt(**user_inputs)
    return prompt


def evaluate(pred, input_text, output_text, eval_aspect):
    """OpenAI API により評価を行う
    Args:
    Returns:
        [dict] 評価結果
        {"reason": "<評価理由>", "grade": <int, 1～5の5段階評価>}
    """
    # `pred` が空の場合は、評点を1にする
    if pred.strip() == "":
        print("回答が空なので１点")
        return {
            "text1": "1",
            "text2": "1",
            "text3": "1",
            "reason": "No response",
        }

    prompt = template_prompt.format(
        input_text=input_text,
        output_text=output_text,
        eval_aspect=eval_aspect,
        pred=pred,
    )

    try:
        chat = [
            {
                "role": "system",
                "content": "You must answer all responses in Japanese.あなたは役に立つ誠実な日本人のアシスタントです。あなたは全ての回答に日本語で答えなければならない。",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        generated_texts = []
        for i in range(3):
            success = False
            retry_count = 0
            while not success:  # 成功するまでリトライ
                if retry_count >= 10:  # 10回リトライであきらめる
                    generated_text = d["content"]
                    generated_texts.append("エラー")
                    break
                url = "http://127.0.0.1:8080/completion"
                payload = {
                    "prompt": prompt,
                    "n_predict": 1,
                    "temperature": 0.7,
                    "cache_prompt": True,
                    "grammar": "root ::= [1-5]",
                }
                json_data = json.dumps(payload)
                r = requests.post(
                    url, data=json_data, headers={"Content-Type": "application/json"}
                )

                d = json.loads(r.content)
                # _validate_schema(d, r)
                generated_text = d["content"]
                generated_texts.append(generated_text)
                print(str(d["timings"]))
                try:
                    int(d["content"])
                except ValueError:
                    print(
                        "数値じゃない出力なのでリトライ:" + d["content"]
                    )  # NGやり直し
                    retry_count = retry_count + 1
                else:
                    success = True  # OK

        timings = d["timings"]
        retobj = {
            "text1": generated_texts[0],
            "text2": generated_texts[1],
            "text3": generated_texts[2],
            "prompt_n": timings["prompt_n"],
            "predicted_n": timings["predicted_n"],
            "prompt_ms": timings["prompt_ms"],
            "predicted_ms": timings["predicted_ms"],
            "prompt_per_second": timings["prompt_per_second"],
            "predicted_per_second": timings["predicted_per_second"],
        }

        return retobj
    except ValueError as e:
        print(f"ValueError occurred: {str(e)}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise


# スコアを計算して返す
def process_results(doc, results):

    print("doc:" + doc["input"])
    print("results:" + results[0])

    ret = evaluate(results[0], doc["input"], doc["output"], doc["eval_aspect"])

    score = (int(ret["text1"]) + int(ret["text2"]) + int(ret["text3"])) / 3.0

    print(
        f"avg: {score}, score1: {ret['text1']}, score2: {ret['text2']}, score3: {ret['text3']}"
    )

    results = {
        "acc": score,
    }
    return results
