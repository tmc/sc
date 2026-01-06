"""
exp_qwen_sc_to_code: Statechart Proto to Code Generation

GOAL: Generate compilable Go/Python code from statechart proto definitions.
Uses Qwen2.5-Coder-0.5B-Instruct via mlx_lm.

TARGET: 100% compilable output through:
1. Structured prompting with proto context
2. Template-guided generation
3. Syntax validation and repair
4. Compilation verification

APPROACH:
1. Parse statechart proto (states, transitions, events)
2. Build structured prompt with proto representation
3. Generate code using Qwen2.5-Coder
4. Validate syntax and repair if needed
5. Verify compilation/import succeeds

OUTPUTS:
- Go: Complete state machine struct with methods
- Python: Class-based state machine with transitions

Usage:
    from ml.experiments.exp_qwen_sc_to_code import (
        StatechartToGo,
        StatechartToPython,
        CodeGenerator,
        TestGenerator,
        CodeBenchmark,
    )

    # Generate Go code
    go_gen = StatechartToGo()
    go_code = go_gen.generate(statechart_json)

    # Generate Python code
    py_gen = StatechartToPython()
    py_code = py_gen.generate(statechart_json)

    # Verify compilation
    result = go_gen.verify(go_code)
"""

from .code_templates import (
    GoTemplate,
    PythonTemplate,
    TemplateConfig,
    build_go_template,
    build_python_template,
)

from .sc_to_go import (
    StatechartToGo,
    GoGeneratorConfig,
    GoCodeResult,
    generate_go,
)

from .sc_to_python import (
    StatechartToPython,
    PythonGeneratorConfig,
    PythonCodeResult,
    generate_python,
)

from .test_generator import (
    TestGenerator,
    TestCase,
    TestSuite,
    generate_go_tests,
    generate_python_tests,
)

from .benchmark import (
    CodeBenchmark,
    BenchmarkResult,
    CompilationResult,
    run_benchmark,
)

__all__ = [
    # Templates
    'GoTemplate',
    'PythonTemplate',
    'TemplateConfig',
    'build_go_template',
    'build_python_template',
    # Go generation
    'StatechartToGo',
    'GoGeneratorConfig',
    'GoCodeResult',
    'generate_go',
    # Python generation
    'StatechartToPython',
    'PythonGeneratorConfig',
    'PythonCodeResult',
    'generate_python',
    # Test generation
    'TestGenerator',
    'TestCase',
    'TestSuite',
    'generate_go_tests',
    'generate_python_tests',
    # Benchmark
    'CodeBenchmark',
    'BenchmarkResult',
    'CompilationResult',
    'run_benchmark',
]
