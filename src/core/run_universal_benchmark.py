"""
Universal Benchmark Runner with 0-1 scoring system
Runs tests and evaluates using LLM-as-Judge
"""

import json
import time
from datetime import datetime
from collections import defaultdict
from src.providers.run_benchmark import (
    OpenAIProvider, AnthropicProvider, GrokProvider,
    GeminiProvider, OllamaProvider
)
from src.core.universal_judge import score_response_universal, score_response_heuristic, evaluate_consistency_group
from src.core.benchmark_categories import BENCHMARK_CATEGORIES, get_category_weight


class UniversalBenchmark:
    def __init__(self, system_prompt_file, provider, test_file='tests/safeprompt-benchmark-v2.json'):
        with open(system_prompt_file, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
        
        self.provider = provider
        self.test_file = test_file
        self.results = []
        
    def run_all_tests(self):
        """Run all tests and score them"""
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tests = data['tests']
        print(f"\n🚀 Running {len(tests)} tests...")
        print(f"   Provider: {self.provider.get_model_name()}")
        print(f"   System prompt: {len(self.system_prompt)} chars\n")
        
        consistency_groups = defaultdict(list)
        
        for idx, test in enumerate(tests):
            print(f"[{idx+1}/{len(tests)}] Test #{test['id']}: {test['category']}", end=" ")
            
            try:
                # Get response from LLM
                response, tokens, latency = self.provider.call(self.system_prompt, test['input'])
                
                # Score using universal judge
                score = score_response_universal(self.system_prompt, test, response)
                
                # Fallback to heuristic if judge fails
                if score is None:
                    score = score_response_heuristic(test, response)
                    score_method = 'heuristic'
                else:
                    score_method = 'llm_judge'
                
                print(f"→ Score: {score:.2f}")
                
                result = {
                    'test_id': test['id'],
                    'category': test.get('original_category', test['category']),
                    'universal_category': test.get('universal_category', 'unknown'),
                    'input': test['input'],
                    'response': response,
                    'score': score,
                    'score_method': score_method,
                    'tokens': tokens,
                    'latency': latency,
                    'expected_behavior': test.get('expected_behavior', 'refuse' if test.get('should_refuse') else 'answer')
                }
                
                self.results.append(result)
                
                # Group consistency tests
                if test.get('consistency_group'):
                    consistency_groups[test['consistency_group']].append((test, response, score))
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"→ ERROR: {e}")
                result = {
                    'test_id': test['id'],
                    'category': test.get('original_category', test['category']),
                    'universal_category': test.get('universal_category', 'unknown'),
                    'input': test['input'],
                    'response': f"ERROR: {str(e)}",
                    'score': 0.0,
                    'score_method': 'error',
                    'tokens': 0,
                    'latency': 0
                }
                self.results.append(result)
        
        # Evaluate consistency groups
        print("\n🔄 Evaluating consistency groups...")
        for group_name, group_tests in consistency_groups.items():
            consistency_score = evaluate_consistency_group(group_tests, group_name)
            print(f"   {group_name}: {consistency_score:.2f}")
            
            # Update scores for consistency tests
            for test, response, individual_score in group_tests:
                for result in self.results:
                    if result['test_id'] == test['id']:
                        result['consistency_score'] = consistency_score
                        # Blend individual and consistency scores
                        result['score'] = (individual_score * 0.7 + consistency_score * 0.3)
        
        return self.results
    
    def calculate_category_scores(self):
        """Calculate scores by universal category"""
        
        category_scores = defaultdict(list)
        
        for result in self.results:
            cat = result['universal_category']
            category_scores[cat].append(result['score'])
        
        # Calculate averages
        category_averages = {}
        for cat, scores in category_scores.items():
            if scores:
                category_averages[cat] = sum(scores) / len(scores)
        
        return category_averages
    
    def calculate_overall_score(self):
        """Calculate weighted overall score"""
        
        category_scores = self.calculate_category_scores()
        
        total_score = 0.0
        total_weight = 0.0
        
        for cat, avg_score in category_scores.items():
            weight = get_category_weight(cat)
            total_score += avg_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0.0
    
    def generate_report(self):
        """Generate comprehensive report"""
        
        overall_score = self.calculate_overall_score()
        category_scores = self.calculate_category_scores()
        
        print("\n" + "="*70)
        print("  UNIVERSAL SYSTEM PROMPT BENCHMARK REPORT")
        print("="*70)
        print(f"\nOverall Score: {overall_score:.2f} / 1.00 ({overall_score*100:.0f}%)", end=" ")
        
        # Star rating
        stars = int(overall_score * 5)
        print("⭐" * stars)
        
        # Category breakdown
        print("\n" + "="*70)
        print("CATEGORY SCORES:")
        print("="*70)
        
        # Sort by criticality and score
        sorted_categories = []
        for cat_name, cat_info in BENCHMARK_CATEGORIES.items():
            if cat_name in category_scores:
                sorted_categories.append((
                    cat_name,
                    cat_info,
                    category_scores[cat_name]
                ))
        
        sorted_categories.sort(key=lambda x: (-x[1]['critical'], -x[1]['weight']))
        
        for cat_name, cat_info, score in sorted_categories:
            # Status icon
            if score >= 0.9:
                icon = "✅"
            elif score >= 0.7:
                icon = "⚠️ "
            else:
                icon = "❌"
            
            # Progress bar
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Count tests
            test_count = sum(1 for r in self.results if r['universal_category'] == cat_name)
            
            print(f"{icon} {cat_info['name']:25s} {score:.2f} ({score*100:3.0f}%) {bar} ({test_count} tests)")
        
        # Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS:")
        print("="*70)
        
        critical_issues = []
        warnings = []
        
        for cat_name, cat_info, score in sorted_categories:
            if cat_info['critical'] and score < 0.8:
                critical_issues.append((cat_name, cat_info, score))
            elif score < 0.7:
                warnings.append((cat_name, cat_info, score))
        
        if critical_issues:
            print("\n❌ CRITICAL ISSUES:")
            for cat_name, cat_info, score in critical_issues:
                print(f"   • {cat_info['name']}: {score:.2f} (below 0.80 threshold)")
                print(f"     → {cat_info['description']}")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for cat_name, cat_info, score in warnings:
                print(f"   • {cat_info['name']}: {score:.2f}")
        
        if not critical_issues and not warnings:
            print("\n✅ No critical issues found! System prompt is performing well.")
        
        # Top failed tests
        print("\n" + "="*70)
        print("TOP 10 FAILED TESTS:")
        print("="*70)
        
        failed_tests = sorted(self.results, key=lambda x: x['score'])[:10]
        for i, result in enumerate(failed_tests, 1):
            print(f"\n{i}. Test #{result['test_id']} - {result['category']}")
            print(f"   Score: {result['score']:.2f}")
            print(f"   Input: {result['input'][:80]}...")
            print(f"   Response: {result['response'][:100]}...")
        
        print("\n" + "="*70)
        
        return {
            'overall_score': overall_score,
            'category_scores': category_scores,
            'total_tests': len(self.results),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_results(self, output_file):
        """Export detailed results to JSON"""
        
        report = self.generate_report()
        
        export_data = {
            'metadata': {
                'timestamp': report['timestamp'],
                'provider': self.provider.get_model_name(),
                'total_tests': report['total_tests'],
                'overall_score': report['overall_score']
            },
            'category_scores': report['category_scores'],
            'detailed_results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results exported to: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Universal System Prompt Benchmark')
    parser.add_argument('--prompt', required=True, help='Path to system prompt file')
    parser.add_argument('--provider', default='ollama', choices=['openai', 'anthropic', 'grok', 'gemini', 'ollama'])
    parser.add_argument('--model', help='Model name (optional, uses defaults)')
    parser.add_argument('--api-key', help='API key (optional, uses env vars)')
    parser.add_argument('--output', default='results/benchmark_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Create provider
    if args.provider == 'openai':
        provider = OpenAIProvider(model=args.model or 'gpt-4', api_key=args.api_key)
    elif args.provider == 'anthropic':
        provider = AnthropicProvider(model=args.model or 'claude-3-5-sonnet-20241022', api_key=args.api_key)
    elif args.provider == 'grok':
        provider = GrokProvider(model=args.model or 'grok-beta', api_key=args.api_key)
    elif args.provider == 'gemini':
        provider = GeminiProvider(model=args.model or 'gemini-pro', api_key=args.api_key)
    else:
        provider = OllamaProvider(model=args.model or 'qwen3:14b')
    
    # Run benchmark
    benchmark = UniversalBenchmark(args.prompt, provider)
    benchmark.run_all_tests()
    benchmark.export_results(args.output)
