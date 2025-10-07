# synthetic_data.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import re

class SyntheticDataGenerator:
    """Generate synthetic training data for language models"""
    
    def __init__(self):
        self.templates = self._create_templates()
    
    def _create_templates(self) -> Dict:
        """Create templates for different data types"""
        return {
            'qa': [
                "Q: {question}\nA: {answer}",
                "Question: {question}\nAnswer: {answer}",
                "{question}\n{answer}"
            ],
            'instruction': [
                "Instruction: {instruction}\nInput: {input}\nOutput: {output}",
                "### Instruction:\n{instruction}\n\n### Response:\n{output}",
                "{instruction}\n\n{output}"
            ],
            'dialogue': [
                "User: {user}\nAssistant: {assistant}",
                "Human: {user}\nAI: {assistant}",
                "{user}\n{assistant}"
            ]
        }
    
    def generate_math_qa(self, n_samples: int = 100) -> List[Dict]:
        """Generate synthetic math Q&A data"""
        data = []
        
        operations = [
            ('add', lambda a, b: a + b, '{a} + {b}'),
            ('subtract', lambda a, b: a - b, '{a} - {b}'),
            ('multiply', lambda a, b: a * b, '{a} × {b}'),
            ('divide', lambda a, b: a // b if b != 0 else 0, '{a} ÷ {b}')
        ]
        
        for _ in range(n_samples):
            a = np.random.randint(1, 100)
            b = np.random.randint(1, 100)
            
            op_name, op_func, op_str = operations[np.random.randint(0, len(operations))]
            
            question = f"What is {op_str.format(a=a, b=b)}?"
            answer = str(op_func(a, b))
            
            data.append({
                'question': question,
                'answer': answer,
                'type': 'math',
                'difficulty': 'easy'
            })
        
        return data
    
    def generate_instruction_data(self, n_samples: int = 100) -> List[Dict]:
        """Generate synthetic instruction-following data"""
        data = []
        
        tasks = [
            {
                'instruction': 'Summarize the following text in one sentence',
                'generator': lambda: self._generate_text_summary()
            },
            {
                'instruction': 'Translate the following to uppercase',
                'generator': lambda: self._generate_case_conversion()
            },
            {
                'instruction': 'Extract numbers from the text',
                'generator': lambda: self._generate_number_extraction()
            }
        ]
        
        for _ in range(n_samples):
            task = tasks[np.random.randint(0, len(tasks))]
            input_text, output_text = task['generator']()
            
            data.append({
                'instruction': task['instruction'],
                'input': input_text,
                'output': output_text,
                'type': 'instruction'
            })
        
        return data
    
    def _generate_text_summary(self) -> Tuple[str, str]:
        """Generate a text and its summary"""
        topics = ['weather', 'food', 'technology', 'sports']
        topic = np.random.choice(topics)
        
        texts = {
            'weather': ("The weather today is sunny with temperatures around 75°F. "
                       "There is a slight breeze from the west.", 
                       "Sunny weather with 75°F temperatures and western breeze."),
            'food': ("The restaurant serves Italian cuisine including pasta and pizza. "
                    "Their specialty is homemade marinara sauce.",
                    "Italian restaurant specializing in pasta, pizza, and marinara sauce."),
            'technology': ("The new smartphone features a 6-inch display and 128GB storage. "
                         "It runs on the latest operating system.",
                         "New smartphone with 6-inch display, 128GB storage, latest OS."),
            'sports': ("The basketball game ended with a score of 98-95. "
                      "The home team won in overtime.",
                      "Home team won basketball game 98-95 in overtime.")
        }
        
        return texts[topic]
    
    def _generate_case_conversion(self) -> Tuple[str, str]:
        """Generate case conversion example"""
        words = ['hello world', 'machine learning', 'data science', 'neural network']
        text = np.random.choice(words)
        return text, text.upper()
    
    def _generate_number_extraction(self) -> Tuple[str, str]:
        """Generate number extraction example"""
        num1 = np.random.randint(1, 100)
        num2 = np.random.randint(1, 100)
        text = f"There are {num1} apples and {num2} oranges in the basket."
        output = f"{num1}, {num2}"
        return text, output
    
    def add_noise(self, data: List[Dict], noise_rate: float = 0.1) -> List[Dict]:
        """Add noise to synthetic data to make it more realistic"""
        noisy_data = []
        
        for item in data:
            if np.random.random() < noise_rate:
                # Add typos
                if 'question' in item:
                    item['question'] = self._add_typos(item['question'])
                if 'answer' in item:
                    # Sometimes wrong answers
                    if np.random.random() < 0.3:
                        item['answer'] = "I don't know"
            
            noisy_data.append(item)
        
        return noisy_data
    
    def _add_typos(self, text: str) -> str:
        """Add random typos"""
        if len(text) < 10 or np.random.random() > 0.3:
            return text
        
        chars = list(text)
        idx = np.random.randint(0, len(chars))
        chars[idx] = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
        return ''.join(chars)
    
    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicates"""
        seen = set()
        unique_data = []
        
        for item in data:
            # Create hash from question/instruction
            key = item.get('question', '') + item.get('instruction', '')
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        return unique_data

def visualize_data_quality():
    """Visualize synthetic vs real data characteristics"""
    generator = SyntheticDataGenerator()
    
    # Generate synthetic data
    synthetic_data = generator.generate_math_qa(1000)
    
    # Analyze characteristics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Length distribution
    ax1 = axes[0, 0]
    question_lengths = [len(d['question'].split()) for d in synthetic_data]
    answer_lengths = [len(d['answer'].split()) for d in synthetic_data]
    
    ax1.hist(question_lengths, bins=20, alpha=0.6, label='Questions', color='blue')
    ax1.hist(answer_lengths, bins=20, alpha=0.6, label='Answers', color='red')
    ax1.set_xlabel('Length (words)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Text Length Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Vocabulary size
    ax2 = axes[0, 1]
    
    all_words = set()
    vocab_growth = []
    
    for i, item in enumerate(synthetic_data):
        words = item['question'].lower().split() + item['answer'].lower().split()
        all_words.update(words)
        if i % 50 == 0:
            vocab_growth.append(len(all_words))
    
    ax2.plot(range(0, len(synthetic_data), 50), vocab_growth, 'g-', linewidth=2)
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Unique Vocabulary Size')
    ax2.set_title('Vocabulary Growth\n(Synthetic data has limited vocab)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Pattern repetition
    ax3 = axes[1, 0]
    
    patterns = {}
    for item in synthetic_data:
        # Extract pattern (e.g., "What is X + Y?")
        pattern = re.sub(r'\d+', 'N', item['question'])
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
    
    labels = [f"Pattern {i+1}" for i in range(len(top_patterns))]
    counts = [count for _, count in top_patterns]
    
    bars = ax3.bar(labels, counts, color='orange', alpha=0.7)
    ax3.set_ylabel('Frequency')
    ax3.set_title('Pattern Repetition\n(High = Less diverse)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Quality metrics
    ax4 = axes[1, 1]
    
    metrics = {
        'Diversity': 30,  # Low for synthetic
        'Naturalness': 50,
        'Correctness': 95,  # High for rule-based
        'Coverage': 40
    }
    
    y_pos = np.arange(len(metrics))
    values = list(metrics.values())
    colors_list = ['red' if v < 50 else 'orange' if v < 70 else 'green' for v in values]
    
    bars = ax4.barh(y_pos, values, color=colors_list, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(metrics.keys())
    ax4.set_xlabel('Score (0-100)')
    ax4.set_title('Synthetic Data Quality Metrics')
    ax4.set_xlim(0, 105)
    ax4.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f' {val}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Synthetic Data Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('synthetic_data_quality.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_training_scenarios():
    """Compare training on different data mixes"""
    scenarios = {
        'Pure Synthetic': {
            'cost': 10,
            'quality': 60,
            'diversity': 40,
            'time_to_generate': 1
        },
        '50% Real + 50% Synthetic': {
            'cost': 55,
            'quality': 80,
            'diversity': 70,
            'time_to_generate': 10
        },
        'Pure Real': {
            'cost': 100,
            'quality': 95,
            'diversity': 100,
            'time_to_generate': 100
        },
        'Real + Augmented': {
            'cost': 70,
            'quality': 90,
            'diversity': 85,
            'time_to_generate': 20
        }
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(scenarios.keys())
    
    # Quality
    ax1 = axes[0, 0]
    quality = [scenarios[n]['quality'] for n in names]
    bars = ax1.bar(range(len(names)), quality, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylabel('Quality Score')
    ax1.set_title('Model Quality')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cost
    ax2 = axes[0, 1]
    cost = [scenarios[n]['cost'] for n in names]
    bars = ax2.bar(range(len(names)), cost, color='red', alpha=0.7)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Relative Cost')
    ax2.set_title('Data Collection Cost')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Diversity
    ax3 = axes[1, 0]
    diversity = [scenarios[n]['diversity'] for n in names]
    bars = ax3.bar(range(len(names)), diversity, color='green', alpha=0.7)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=15, ha='right')
    ax3.set_ylabel('Diversity Score')
    ax3.set_title('Data Diversity')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Time to generate
    ax4 = axes[1, 1]
    time = [scenarios[n]['time_to_generate'] for n in names]
    bars = ax4.bar(range(len(names)), time, color='orange', alpha=0.7)
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=15, ha='right')
    ax4.set_ylabel('Relative Time')
    ax4.set_title('Time to Generate Dataset')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Training Data Scenarios Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data_scenarios.png', dpi=150, bbox_inches='tight')
    plt.show()

def model_collapse_demonstration():
    """Show risk of model collapse with pure synthetic data"""
    generations = 5
    
    # Simulate quality degradation
    quality_pure_synthetic = [100]
    quality_mixed = [100]
    quality_curated = [100]
    
    for gen in range(1, generations):
        # Pure synthetic degrades quickly
        quality_pure_synthetic.append(quality_pure_synthetic[-1] * 0.7)
        
        # Mixed data degrades slowly
        quality_mixed.append(quality_mixed[-1] * 0.95)
        
        # Curated stays stable
        quality_curated.append(quality_curated[-1] * 0.98)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Quality over generations
    x = range(generations)
    ax1.plot(x, quality_pure_synthetic, 'r-o', linewidth=2, 
            label='Pure Synthetic', markersize=8)
    ax1.plot(x, quality_mixed, 'orange', linewidth=2, 
            label='50% Synthetic', marker='s', markersize=8)
    ax1.plot(x, quality_curated, 'g-^', linewidth=2, 
            label='Curated Synthetic', markersize=8)
    
    ax1.axhline(y=80, color='red', linestyle='--', alpha=0.3, label='Critical threshold')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Model Quality (%)')
    ax1.set_title('Model Collapse Over Generations\n(Training on own outputs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Diversity loss
    diversity_pure = [100, 70, 40, 20, 10]
    diversity_mixed = [100, 90, 80, 72, 65]
    diversity_curated = [100, 95, 90, 87, 85]
    
    ax2.plot(x, diversity_pure, 'r-o', linewidth=2, markersize=8)
    ax2.plot(x, diversity_mixed, 'orange', linewidth=2, marker='s', markersize=8)
    ax2.plot(x, diversity_curated, 'g-^', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Output Diversity (%)')
    ax2.set_title('Diversity Loss\n(Mode collapse)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('model_collapse.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_experiments():
    print("Synthetic Data Generation Analysis")
    print("=" * 50)
    
    generator = SyntheticDataGenerator()
    
    print("\n1. Generating synthetic data...")
    math_data = generator.generate_math_qa(100)
    instruction_data = generator.generate_instruction_data(100)
    print(f"   Generated {len(math_data)} math Q&A pairs")
    print(f"   Generated {len(instruction_data)} instruction examples")
    
    print("\n2. Analyzing data quality...")
    visualize_data_quality()
    print("   Saved: synthetic_data_quality.png")
    
    print("\n3. Comparing training scenarios...")
    compare_training_scenarios()
    print("   Saved: data_scenarios.png")
    
    print("\n4. Demonstrating model collapse risk...")
    model_collapse_demonstration()
    print("   Saved: model_collapse.png")
    
    # Show example
    print("\n5. Example synthetic data:")
    print("-" * 50)
    for i in range(3):
        item = math_data[i]
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}\n")
    
    print("=" * 50)
    print("KEY INSIGHTS:")
    print("1. Synthetic data: Fast and cheap but limited diversity")
    print("2. Model collapse: Training on own outputs degrades quality")
    print("3. Best practice: Mix synthetic with real (50/50 or less)")
    print("4. Curation is critical: Filter low-quality synthetic data")
    print("5. Augmentation > pure generation for most tasks")
    print("6. Use synthetic for: math, code, structured tasks")
    print("\nProduction: Generate synthetic, but always mix with real data")

if __name__ == "__main__":
    run_experiments()
