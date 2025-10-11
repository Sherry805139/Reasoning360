from verl.utils.reward_score.prime_math.grader import math_equal
from verl.utils.reward_score import math
import re
from typing import Tuple
import os, requests


def _check_single_answer(answer: str, ground_truth: str) -> bool:
    try:
        nanswer = answer.replace(",", "").replace("%", " / 100").replace("$", "").replace(":", "/").replace("\\", "")
        nanswer = float(eval(nanswer))
        return math_equal(nanswer, ground_truth, tolerance=1e-3)
    except:
        # If the answer is not a number, use the original answer for full string match
        return math.is_equiv(answer, ground_truth)

def drop_latex_text(answer: str) -> str:
    # Remove \\text{} from "20 \\text{to} 39". There could be multiple \\text{} in the answer.
    # Replace \text{something} with something
    answer = re.sub(r'\\\\text\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\\\', r'', answer)
    return answer

def combine(input1):
    if "\\boxed{" in input1:
        return "Here are the suggestions from others as additional information:\n"+"Suggestion 1:\n"+input1.split("\\boxed{")[0]+"\\boxed{"+input1.split("\\boxed{")[1].split("}")[0]+"}\n"
    else:
        return ""

def _llm_attacked(question: str, ground_truth: str):
    url_base = os.getenv("STEM_LLM_JUDGE_URL")
    if not url_base:
        raise EnvironmentError("STEM_LLM_JUDGE_URL not set")
    url = url_base.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": "/home/hmpiao/adv_reason/Reasoning360/checkpoints/Reasoning360-1.7B/--Qwen3-1.7B-Base-think-e6-s1-direct-frompretrain-0.2/global_step_40/actor/Qwen_Qwen3-1.7B-think-e6-s1-step40-direct-frompretrain-0.2",
        "messages": [{"role": "user", "content": question}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 1.0
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
    except Exception as e:
        print(f"Timeout in vllm: {e}")
        score_attacked = 0
        return score_attacked
    resp.raise_for_status()
    data = resp.json()

    model_output = data["choices"][0]["message"]["content"]
    model_output_attacked = str(model_output).lower()
    ground_truth_attacked = str(ground_truth).lower()
    solution_str_attacked = model_output_attacked.split("</think>")[-1]
    answer_str_attacked = math.last_boxed_only_string(solution_str_attacked)
    if answer_str_attacked is not None:
        answer_attacked = math.remove_boxed(answer_str_attacked)
        answer_attacked = drop_latex_text(answer_attacked)
    else:
        answer_attacked = solution_str_attacked

    # print(f">>> {answer}, {ground_truth}")
    if "|" not in ground_truth_attacked:
        # Single numeric answer
        score_attacked = _check_single_answer(answer_attacked, ground_truth_attacked)
    else:
        # Multiple answers, in format "ans1|ans2|ans3"
        try:
            ground_truth_attacked = sorted([ans.strip() for ans in ground_truth_attacked.split("|")])
            answer_attacked = sorted([ans.strip() for ans in answer_attacked.split("|")])
            if len(ground_truth_attacked) != len(answer_attacked):
                score_attacked = 0
            else:
                score_attacked = 1
                for gt, res in zip(ground_truth_attacked, answer_attacked):
                    score_attacked = _check_single_answer(res, gt)
                    if not score_attacked:
                        break
        except Exception as e:
            print(f"Error postprocessing ground truth or response in tablereason.py: {e}")
            score_attacked = 0
    return score_attacked


def compute_score_adver(data_source: str,
                  model_output: str,
                  ground_truth: str,
                  extra_info: dict) -> bool:
    #计算生成答案的正确性
    model_output_attacker = str(model_output).lower()
    ground_truth_attacker = str(ground_truth).lower()

    step_in = False
    obj_in = False
    res_in = False
    if "<step>" in model_output_attacker and "</step>" in model_output_attacker:
        step_in = True

    if "<obj>" in model_output_attacker and "</obj>" in model_output_attacker:
        obj_in = True

    if "<res>" in model_output_attacker and "</res>" in model_output_attacker:
        res_in = True

    solution_str_attacker = model_output_attacker.split("</think>")[-1]
    answer_str_attacker = math.last_boxed_only_string(solution_str_attacker)
    if answer_str_attacker is not None:
        answer_attacker = math.remove_boxed(answer_str_attacker)
        answer_attacker = drop_latex_text(answer_attacker)
    else:
        answer_attacker = solution_str_attacker

    # print(f">>> {answer}, {ground_truth}")
    if "|" not in ground_truth_attacker:
        # Single numeric answer
        score_attacker = _check_single_answer(answer_attacker, ground_truth_attacker)
    else:
        # Multiple answers, in format "ans1|ans2|ans3"
        try:
            ground_truth_attacker = sorted([ans.strip() for ans in ground_truth_attacker.split("|")])
            answer_attacker = sorted([ans.strip() for ans in answer_attacker.split("|")])
            if len(ground_truth_attacker) != len(answer_attacker):
                score_attacker = 0
            else:
                score_attacker = 1
                for gt, res in zip(ground_truth_attacker, answer_attacker):
                    score_attacker = _check_single_answer(res, gt)
                    if not score_attacker:
                        break
        except Exception as e:
            print(f"Error postprocessing ground truth or response in tablereason.py: {e}")
            score_attacker = 0
    #计算被攻击者正确性
    prompt_before = extra_info["prompt"]['content']
    try:
        score_actor_before = _llm_attacked(prompt_before, ground_truth)
    except Exception as e:
        score_actor_before = 0
    try:
        combine_str = combine(str(model_output))
        prompt_after = prompt_before.split("Please output the final answer")[0]+combine_str+"Please output the final answer"+prompt_before.split("Please output the final answer")[1]
        if len(combine_str) == 0:
            score_actor_after = 0
        else:
            score_actor_after = _llm_attacked(prompt_after, ground_truth)
    except Exception as e:
        score_actor_after = 0
    
    #for the learning to accept attacker: 生成对的但模型接受不了的答案
    #if score_attacker == 1 and score_actor_after == 0:
        #return {"score": 1, "acc": 1}
    #else:
        #return {"score": 0, "acc": 0}
    #for the learning to reject attacker: 生成错的并且把模型带偏的答案
    if score_attacker == 0 and score_actor_before == 1 and score_actor_after == 0 and step_in and res_in and obj_in:
        return {"score": 1, "acc": 1}
    elif score_attacker == 1 and step_in and res_in and obj_in:
        return {"score": 0.5, "acc": 0.5}
    else:
        return {"score": 0, "acc": 0}
    

def compute_score(model_output: str, ground_truth: str, extra_info: any = None) -> bool:
    model_output = str(model_output).lower()
    ground_truth = str(ground_truth).lower()
    
    solution_str = model_output.split("</think>")[-1]
    answer_str = math.last_boxed_only_string(solution_str)
    if answer_str is not None:
        answer = math.remove_boxed(answer_str)
        answer = drop_latex_text(answer)
    else:
        answer = solution_str

    # print(f">>> {answer}, {ground_truth}")
    if "|" not in ground_truth:
        # Single numeric answer
        score = _check_single_answer(answer, ground_truth)
    else:
        # Multiple answers, in format "ans1|ans2|ans3"
        try:
            ground_truth = sorted([ans.strip() for ans in ground_truth.split("|")])
            answer = sorted([ans.strip() for ans in answer.split("|")])
            if len(ground_truth) != len(answer):
                score = 0
            else:
                score = 1
                for gt, res in zip(ground_truth, answer):
                    score = _check_single_answer(res, gt)
                    if not score:
                        break
        except Exception as e:
            print(f"Error postprocessing ground truth or response in tablereason.py: {e}")
            return {"score": 0, "acc": 0}

    return {"score": score, "acc": score}

def compute_score_cot_step(model_output: str, ground_truth: str, extra_info: any = None) -> bool:
    model_output = str(model_output).lower()
    ground_truth = str(ground_truth).lower()

    cot_in = False
    step_in = False
    obj_in = False
    res_in = False
    if "</think>" in model_output:
        cot_in = True

    if "<step>" in model_output and "</step>" in model_output:
        step_in = True

    if "<obj>" in model_output and "</obj>" in model_output:
        obj_in = True

    if "<res>" in model_output and "</res>" in model_output:
        res_in = True
    
    solution_str = model_output.split("</think>")[-1]
    answer_str = math.last_boxed_only_string(solution_str)
    if answer_str is not None:
        answer = math.remove_boxed(answer_str)
        answer = drop_latex_text(answer)
    else:
        answer = solution_str

    # print(f">>> {answer}, {ground_truth}")
    if "|" not in ground_truth:
        # Single numeric answer
        score = _check_single_answer(answer, ground_truth)
    else:
        # Multiple answers, in format "ans1|ans2|ans3"
        try:
            ground_truth = sorted([ans.strip() for ans in ground_truth.split("|")])
            answer = sorted([ans.strip() for ans in answer.split("|")])
            if len(ground_truth) != len(answer):
                score = 0
            else:
                score = 1
                for gt, res in zip(ground_truth, answer):
                    score = _check_single_answer(res, gt)
                    if not score:
                        break
        except Exception as e:
            print(f"Error postprocessing ground truth or response in tablereason.py: {e}")
            return {"score": 0, "acc": 0}
        
    if score == 1 and step_in and obj_in and res_in:
        score = 1
    elif score == 1:
        score = 0.5
    else:
        score = 0
    return {"score": score, "acc": score}


def compute_score_cot_step_judge(model_output: str, ground_truth: str, extra_info: any = None) -> bool:
    model_output = str(model_output).lower()
    ground_truth = str(ground_truth).lower()

    cot_in = False
    step_in = False
    obj_in = False
    res_in = False
    judge_in = True
    detect = False
    change_answer = False
    if "</think>" in model_output:
        cot_in = True

    if "<step>" in model_output and "</step>" in model_output:
        step_in = True

    if "<obj>" in model_output and "</obj>" in model_output:
        obj_in = True

    if "<res>" in model_output and "</res>" in model_output:
        res_in = True

    if extra_info["judge"] == "True" and not ("<judge>" in model_output and "</judge>" in model_output):
        judge_in = False

    if extra_info["judge"] == "False" and ("<judge>" in model_output or "</judge>" in model_output):
        judge_in = False
    
    solution_str = model_output.split("</think>")[-1]
    answer_str = math.last_boxed_only_string(solution_str)
    if answer_str is not None:
        answer = math.remove_boxed(answer_str)
        answer = drop_latex_text(answer)
    else:
        answer = solution_str

    answer_sug = answer
    # print(f">>> {answer}, {ground_truth}")
    if "|" not in ground_truth:
        # Single numeric answer
        score = _check_single_answer(answer, ground_truth)
    else:
        # Multiple answers, in format "ans1|ans2|ans3"
        try:
            ground_truth = sorted([ans.strip() for ans in ground_truth.split("|")])
            answer = sorted([ans.strip() for ans in answer.split("|")])
            if len(ground_truth) != len(answer):
                score = 0
            else:
                score = 1
                for gt, res in zip(ground_truth, answer):
                    score = _check_single_answer(res, gt)
                    if not score:
                        break
        except Exception as e:
            print(f"Error postprocessing ground truth or response in tablereason.py: {e}")
            return {"score": 0, "acc": 0}

    if extra_info["rs"] == "True":    
        ground_truth_sug = extra_info["ground_truth_sug"]
        if "|" not in ground_truth_sug:
            # Single numeric answer
            change_answer = _check_single_answer(answer_sug, ground_truth_sug)
        else:
            # Multiple answers, in format "ans1|ans2|ans3"
            try:
                ground_truth_sug = sorted([ans.strip() for ans in ground_truth_sug.split("|")])
                answer_sug = sorted([ans.strip() for ans in answer_sug.split("|")])
                if len(ground_truth_sug) != len(answer_sug):
                    change_answer = 1
                else:
                    change_answer = 1
                    for gt, res in zip(ground_truth_sug, answer_sug):
                        change_answer = _check_single_answer(res, gt)
                        if not change_answer:
                            break
            except Exception as e:
                print(f"Error postprocessing ground truth or response in tablereason.py: {e}")
                change_answer = 1
        
        
    if extra_info["judge"] == "False":
        detect = True
    else:
        if "<judge>" in model_output and "</judge>" in model_output:
            if (extra_info["rs"] == "True" and "incorrect because: " in model_output.split("</judge>")[0] and " correct" not in model_output.split("</judge>")[0] and not change_answer) or (extra_info["rs"] == "False" and " correct because: " in model_output.split("</judge>")[0] and "incorrect" not in model_output.split("</judge>")[0]):
                detect = True
            else:
                detect = False
        else:
            detect = False
        

    
    if score == 1 and step_in and obj_in and res_in and judge_in and detect:
        score = 1
    elif score == 1 and step_in and obj_in and res_in and judge_in:
        score = 0.6   
    elif step_in and obj_in and res_in and judge_in and detect and extra_info["rs"] == "True":
        score = 0.6
    elif score == 1:
        score = 0.4
    else:
        score = 0
    return {"score": score, "acc": score}
