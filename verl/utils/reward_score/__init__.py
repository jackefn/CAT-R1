# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    res = qa_em_and_format.compute_score_format_answer(solution_str, ground_truth)

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    

def _default_compute_score_format(data_source, solution_str, extra_info=None):
    from . import qa_em_and_format
    print("start compute_score_format")
    res = qa_em_and_format.compute_score_format(solution_str)
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score_answer_em(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    res = qa_em_and_format.compute_score_em(solution_str, ground_truth)
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score_answer_f1(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    res = qa_em_and_format.compute_score_f1(solution_str, ground_truth)
    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    res = qa_em_and_format.compute_score_format_answer(solution_str, ground_truth)

    
    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])