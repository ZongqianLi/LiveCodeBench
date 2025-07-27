import json
import sys
import os

# 添加当前路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入LiveCodeBench现有模块
from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.utils.extraction_utils import extract_code, LMStyle

# 测试数据1 - AtCoder题目（标准输入输出）
problem1 = {
    'question_content': '''As KEYENCE headquarters have more and more workers, they decided to divide the departments in the headquarters into two groups and stagger their lunch breaks.
KEYENCE headquarters have N departments, and the number of people in the i-th department (1≤i≤N) is K_i.
When assigning each department to Group A or Group B, having each group take lunch breaks at the same time, and ensuring that the lunch break times of Group A and Group B do not overlap, find the minimum possible value of the maximum number of people taking a lunch break at the same time.''',
    'public_test_cases': [
        {"input": "5\n2 3 5 10 12\n", "output": "17\n", "testtype": "stdin"},
        {"input": "2\n1 1\n", "output": "1\n", "testtype": "stdin"},
        {"input": "6\n22 25 26 45 22 31\n", "output": "89\n", "testtype": "stdin"}
    ],
    'platform': 'atcoder'
}

# 测试数据2 - LeetCode题目（函数调用）
problem2 = {
    'question_content': '''You are given an integer array nums. You need to ensure that the elements in the array are distinct. To achieve this, you can perform the following operation any number of times:
Remove 3 elements from the beginning of the array. If the array has fewer than 3 elements, remove all remaining elements.''',
    'public_test_cases': [
        {"input": "[1, 2, 3, 4, 2, 3, 3, 5, 7]", "output": "2", "testtype": "functional"},
        {"input": "[4, 5, 6, 4, 4]", "output": "2", "testtype": "functional"},
        {"input": "[6, 7, 8, 9]", "output": "0", "testtype": "functional"}
    ],
    'func_name': 'minimumOperations',
    'platform': 'leetcode'
}

def test_code_solution(problem, model_output):
    """
    测试从模型输出中提取的代码
    
    Args:
        problem: 问题数据，包含测试用例
        model_output: 模型的原始输出文本
    
    Returns:
        测试结果字典
    """
    print(f"Testing solution for {problem['platform']} problem...")
    
    # 1. 从模型输出中提取代码
    extracted_code = extract_code(model_output, LMStyle.OpenAIChat)
    print(f"Extracted code:\n{extracted_code}\n")
    
    # 2. 准备测试样本
    test_sample = prepare_test_sample(problem)
    
    # 3. 运行测试
    result, metadata = run_test(test_sample, test=extracted_code, debug=True, timeout=6)
    
    # 4. 分析结果
    analyze_result(result, metadata, problem['public_test_cases'])
    
    return {
        'code': extracted_code,
        'results': result,
        'metadata': metadata,
        'passed': all(r > 0 for r in result) if result else False
    }

def prepare_test_sample(problem):
    """准备测试样本"""
    test_cases = problem['public_test_cases']
    
    if problem['platform'] == 'leetcode':
        # 函数调用模式
        inputs = [tc['input'] for tc in test_cases]
        outputs = [tc['output'] for tc in test_cases]
        func_name = problem.get('func_name', 'solution')
        
        sample = {
            "input_output": json.dumps({
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": func_name
            })
        }
    else:
        # 标准输入输出模式
        inputs = [tc['input'] for tc in test_cases]
        outputs = [tc['output'] for tc in test_cases]
        
        sample = {
            "input_output": json.dumps({
                "inputs": inputs,
                "outputs": outputs
            })
        }
    
    return sample

def analyze_result(result, metadata, test_cases):
    """分析测试结果"""
    print(f"Test Results: {result}")
    print(f"Metadata: {metadata}")
    
    if result:
        passed = sum(1 for r in result if r > 0)
        total = len(result)
        print(f"Passed: {passed}/{total} test cases")
        
        for i, (r, tc) in enumerate(zip(result, test_cases)):
            status = "✓ PASS" if r > 0 else "✗ FAIL"
            print(f"  Test {i+1}: {status}")
            if r <= 0:
                print(f"    Input: {tc['input']}")
                print(f"    Expected: {tc['output']}")
    else:
        print("No results - compilation or runtime error")

if __name__ == "__main__":
    # 示例1: 测试AtCoder问题的解决方案
    atcoder_solution = '''
def solve():
    n = int(input())
    k = list(map(int, input().split()))
    
    total = sum(k)
    min_max = float('inf')
    
    # 尝试所有可能的分组
    for mask in range(1 << n):
        group_a = sum(k[i] for i in range(n) if mask & (1 << i))
        group_b = total - group_a
        min_max = min(min_max, max(group_a, group_b))
    
    print(min_max)

solve()
'''
    
    print("=== Testing AtCoder Solution ===")
    result1 = test_code_solution(problem1, f"```python\n{atcoder_solution}\n```")
    
    # 示例2: 测试LeetCode问题的解决方案
    leetcode_solution = '''
class Solution:
    def minimumOperations(self, nums):
        operations = 0
        while len(nums) > len(set(nums)):
            if len(nums) >= 3:
                nums = nums[3:]
            else:
                nums = []
            operations += 1
        return operations
'''
    
    print("\n=== Testing LeetCode Solution ===")
    result2 = test_code_solution(problem2, f"```python\n{leetcode_solution}\n```")
    
    # 输出总结
    print("\n=== Summary ===")
    print(f"AtCoder solution: {'PASSED' if result1['passed'] else 'FAILED'}")
    print(f"LeetCode solution: {'PASSED' if result2['passed'] else 'FAILED'}")