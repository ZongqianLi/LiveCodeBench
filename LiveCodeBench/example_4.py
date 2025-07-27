"""
20250727
test cases是从本地加载的

0250727
支持设置batch size和num_workers 提高了代码测试的速度

20250726
如果将livecodebench的test cases转变为assert格式
则会导致测试出来的结果偏低
比如qwen 3 4b thinking用这个文件测约是55% 接近官方给出的值 但是转变为assert后测只有约50%
这个文件是用官方的代码进行测试
只需要把这个文件放在从github上下载的LiveCodeBench包的根目录下即可
输入时val后保存的详细的模型输出结果
从模型输出中提取代码 然后测试
输出结果
"""

import json
import sys
import os
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 添加当前路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入LiveCodeBench现有模块
from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.utils.extraction_utils import extract_code, LMStyle

def load_test_cases_from_local(data_source, test_cases_dir="/mnt/nvme2n1/sglang/data/original_test_cases/LiveCodeBench"):
    """
    从本地文件加载测试用例
    
    Args:
        data_source: 数据源标识符
        test_cases_dir: 测试用例目录路径
    
    Returns:
        dict: 包含test_cases和func_name的数据
    """
    test_case_file = os.path.join(test_cases_dir, f"{data_source}.json")
    
    if not os.path.exists(test_case_file):
        raise ValueError(f"Test case file not found: {test_case_file}")
    
    with open(test_case_file, 'r', encoding='utf-8') as f:
        test_case_data = json.load(f)
    
    return test_case_data

def prepare_test_sample_from_local(test_case_data):
    """从本地测试用例数据准备测试样本"""
    test_cases = test_case_data['test_cases']
    func_name = test_case_data['func_name']
    
    if func_name:
        # 函数调用模式 (LeetCode)
        inputs = [tc['input'] for tc in test_cases]
        outputs = [tc['output'] for tc in test_cases]
        
        sample = {
            "input_output": json.dumps({
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": func_name
            })
        }
    else:
        # 标准输入输出模式 (AtCoder/CodeForces)
        inputs = [tc['input'] for tc in test_cases]
        outputs = [tc['output'] for tc in test_cases]
        
        sample = {
            "input_output": json.dumps({
                "inputs": inputs,
                "outputs": outputs
            })
        }
    
    return sample, test_cases

def test_single_solution_worker(args):
    """
    并行工作函数 - 测试单个解决方案
    """
    data_source, model_output_raw, test_cases_dir, detailed = args
    
    # 从本地加载测试用例
    try:
        test_case_data = load_test_cases_from_local(data_source, test_cases_dir)
    except Exception as e:
        return {
            'data_source': data_source,
            'error_type': 'load_test_cases_error',
            'error_message': str(e),
            'successfully_tested': False
        }
    
    # 提取代码
    extracted_code = extract_code(model_output_raw, LMStyle.OpenAIChat)
    if len(extracted_code) == 0:
        return {
            'data_source': data_source,
            'error_type': 'code_extraction_error',
            'error_message': 'No code extracted from model output',
            'successfully_tested': False
        }
    
    # 准备测试样本
    try:
        test_sample, all_test_cases = prepare_test_sample_from_local(test_case_data)
    except Exception as e:
        return {
            'data_source': data_source,
            'error_type': 'test_preparation_error',
            'error_message': str(e),
            'successfully_tested': False
        }
    
    # 运行测试
    try:
        result, metadata = run_test(test_sample, test=extracted_code, debug=False, timeout=6)
        
        # 分析结果
        if result:
            passed = sum(1 for r in result if r > 0)
            total = len(result)
            pass_rate = passed / total if total > 0 else 0
            all_passed = (passed == total)
            
            return {
                'data_source': data_source,
                'total_tests': total,
                'passed_tests': passed,
                'pass_rate': pass_rate,
                'all_passed': all_passed,
                'results': result,
                'metadata': metadata,
                'extracted_code': extracted_code,
                'successfully_tested': True
            }
        else:
            return {
                'data_source': data_source,
                'total_tests': 0,
                'passed_tests': 0,
                'pass_rate': 0,
                'all_passed': False,
                'results': None,
                'metadata': metadata,
                'extracted_code': extracted_code,
                'successfully_tested': True
            }
            
    except Exception as e:
        return {
            'data_source': data_source,
            'error_type': 'test_execution_error',
            'error_message': str(e),
            'successfully_tested': False
        }

def test_single_solution(data_source, model_output_raw, test_cases_dir, detailed=False):
    """
    测试单个解决方案
    """
    print(f"\n{'='*80}")
    print(f"Testing: {data_source}")
    
    # 从本地加载测试用例
    try:
        test_case_data = load_test_cases_from_local(data_source, test_cases_dir)
        print(f"Loaded {test_case_data['total_cases']} test cases")
    except Exception as e:
        print(f"Error loading test cases: {e}")
        return {
            'data_source': data_source,
            'error_type': 'load_test_cases_error',
            'error_message': str(e),
            'successfully_tested': False
        }
    
    # 提取代码
    extracted_code = extract_code(model_output_raw, LMStyle.OpenAIChat)
    print(f"Extracted code length: {len(extracted_code)} characters")
    if len(extracted_code) == 0:
        print("Warning: No code extracted from model output")
        return {
            'data_source': data_source,
            'error_type': 'code_extraction_error',
            'error_message': 'No code extracted from model output',
            'successfully_tested': False
        }
    
    # 准备测试样本
    try:
        test_sample, all_test_cases = prepare_test_sample_from_local(test_case_data)
    except Exception as e:
        print(f"Error preparing test sample: {e}")
        return {
            'data_source': data_source,
            'error_type': 'test_preparation_error',
            'error_message': str(e),
            'successfully_tested': False
        }
    
    print(f"Using {len(all_test_cases)} test cases")
    
    # 运行测试
    try:
        result, metadata = run_test(test_sample, test=extracted_code, debug=False, timeout=6)
        
        # 分析结果
        if result:
            passed = sum(1 for r in result if r > 0)
            total = len(result)
            pass_rate = passed / total if total > 0 else 0
            all_passed = (passed == total)
            print(f"Result: {passed}/{total} tests passed ({pass_rate:.2%})")
            
            return {
                'data_source': data_source,
                'total_tests': total,
                'passed_tests': passed,
                'pass_rate': pass_rate,
                'all_passed': all_passed,
                'results': result,
                'metadata': metadata,
                'extracted_code': extracted_code,
                'successfully_tested': True
            }
        else:
            print("Result: Compilation or runtime error")
            return {
                'data_source': data_source,
                'total_tests': 0,
                'passed_tests': 0,
                'pass_rate': 0,
                'all_passed': False,
                'results': None,
                'metadata': metadata,
                'extracted_code': extracted_code,
                'successfully_tested': True
            }
            
    except Exception as e:
        print(f"Error running test: {e}")
        return {
            'data_source': data_source,
            'error_type': 'test_execution_error',
            'error_message': str(e),
            'successfully_tested': False
        }

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Test LiveCodeBench solutions from JSON file')
    parser.add_argument('json_file', help='Path to JSON file containing model outputs')
    parser.add_argument('--output', help='Path to save results (optional)')
    parser.add_argument('--detailed', action='store_true', help='Run individual test cases on failures')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of parallel workers (default: 12)')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing (default: 100)')
    parser.add_argument('--test_cases_dir', default="/mnt/nvme2n1/sglang/data/original_test_cases/LiveCodeBench", 
                       help='Directory containing test case files (default: /mnt/nvme2n1/sglang/data/original_test_cases/LiveCodeBench)')
    args = parser.parse_args()
    
    # 读取JSON文件
    print(f"Loading solutions from {args.json_file}...")
    with open(args.json_file, 'r') as f:
        solutions = json.load(f)
    
    print(f"Found {len(solutions)} solutions to test")
    print(f"Test cases directory: {args.test_cases_dir}")
    
    # 准备有效的解决方案
    valid_solutions = []
    invalid_results = []
    
    for solution in solutions:
        if 'data_source' not in solution or 'model_output_raw' not in solution:
            invalid_results.append({
                'data_source': solution.get('data_source', 'unknown'),
                'error_type': 'missing_fields',
                'error_message': 'Missing required fields (data_source or model_output_raw)',
                'successfully_tested': False
            })
        else:
            valid_solutions.append(solution)
    
    print(f"Valid solutions to test: {len(valid_solutions)}")
    print(f"Using {args.num_workers} workers with batch size {args.batch_size}")
    
    # 并行测试所有解决方案
    all_results = invalid_results.copy()
    successfully_tested = 0
    all_passed_count = 0
    
    # 准备参数列表
    test_args = [
        (solution['data_source'], solution['model_output_raw'], args.test_cases_dir, args.detailed)
        for solution in valid_solutions
    ]
    
    # 分批处理以控制内存使用
    for batch_start in range(0, len(test_args), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(test_args))
        batch_args = test_args[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//args.batch_size + 1}/{(len(test_args) + args.batch_size - 1)//args.batch_size}")
        print(f"Batch size: {len(batch_args)} (items {batch_start+1}-{batch_end})")
        
        # 并行执行当前批次
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(test_single_solution_worker, arg): i for i, arg in enumerate(batch_args)}
            
            # 使用进度条显示批次进度
            with tqdm(total=len(batch_args), desc=f"Batch {batch_start//args.batch_size + 1}") as pbar:
                batch_results = [None] * len(batch_args)
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        batch_results[idx] = result
                    except Exception as e:
                        # 处理执行异常
                        data_source = batch_args[idx][0]
                        batch_results[idx] = {
                            'data_source': data_source,
                            'error_type': 'execution_exception',
                            'error_message': str(e),
                            'successfully_tested': False
                        }
                    pbar.update(1)
        
        # 将批次结果添加到总结果中
        all_results.extend(batch_results)
        
        # 统计当前批次结果
        for result in batch_results:
            if result and result.get('successfully_tested', False):
                successfully_tested += 1
                if result.get('all_passed', False):
                    all_passed_count += 1
    
    # 统计未成功测试的情况
    failed_results = [r for r in all_results if not r.get('successfully_tested', False)]
    failed_by_type = {}
    for failed in failed_results:
        error_type = failed.get('error_type', 'unknown')
        if error_type not in failed_by_type:
            failed_by_type[error_type] = 0
        failed_by_type[error_type] += 1
    
    # 输出总结
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total solutions: {len(solutions)}")
    print(f"Successfully tested: {successfully_tested}")
    print(f"Failed to test: {len(solutions) - successfully_tested}")
    
    if failed_by_type:
        print("Failed test breakdown:")
        for error_type, count in failed_by_type.items():
            print(f"  {error_type}: {count}")
    
    # 修改后的Average pass rate计算
    average_pass_rate = all_passed_count / len(solutions) if len(solutions) > 0 else 0
    print(f"Average pass rate (all correct): {all_passed_count}/{len(solutions)} = {average_pass_rate:.2%}")
    
    # 统计成功测试中的pass rate
    successfully_tested_results = [r for r in all_results if r.get('successfully_tested', False)]
    if successfully_tested_results:
        total_pass_rate = sum(r.get('pass_rate', 0) for r in successfully_tested_results) / len(successfully_tested_results)
        print(f"Average pass rate (among successfully tested): {total_pass_rate:.2%}")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # 计算并输出总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    import os
    os.system('sudo chmod -R 777 /mnt/nvme2n1/sglang/models/validation_results/')
    main()