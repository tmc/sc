"""
Unit tests for statechart validation functionality.

This module tests the validation utilities for statechart definitions,
including local validation rules and error detection.
"""

import pytest

from statecharts.validation import (
    LocalValidator,
    ValidationResult,
    Violation,
    Severity,
    RuleId,
    validate_statechart,
)
from statecharts.core import StateType
from statecharts.factory import (
    basic_state,
    normal_state,
    parallel_state,
    transition,
    event,
    statechart,
)


class TestSeverity:
    """Test Severity enum."""
    
    def test_severity_values(self):
        """Test that Severity enum has correct values."""
        assert Severity.UNSPECIFIED.name == "UNSPECIFIED"
        assert Severity.INFO.name == "INFO"
        assert Severity.WARNING.name == "WARNING"
        assert Severity.ERROR.name == "ERROR"


class TestRuleId:
    """Test RuleId enum."""
    
    def test_rule_id_values(self):
        """Test that RuleId enum has correct values."""
        assert RuleId.UNSPECIFIED.name == "UNSPECIFIED"
        assert RuleId.UNIQUE_STATE_LABELS.name == "UNIQUE_STATE_LABELS"
        assert RuleId.SINGLE_DEFAULT_CHILD.name == "SINGLE_DEFAULT_CHILD"
        assert RuleId.BASIC_HAS_NO_CHILDREN.name == "BASIC_HAS_NO_CHILDREN"
        assert RuleId.COMPOUND_HAS_CHILDREN.name == "COMPOUND_HAS_CHILDREN"


class TestViolation:
    """Test Violation class."""
    
    def test_violation_creation(self):
        """Test creating a Violation."""
        violation = Violation(
            rule=RuleId.UNIQUE_STATE_LABELS,
            severity=Severity.ERROR,
            message="Duplicate state label found",
            xpath=["root", "child"]
        )
        
        assert violation.rule == RuleId.UNIQUE_STATE_LABELS
        assert violation.severity == Severity.ERROR
        assert violation.message == "Duplicate state label found"
        assert violation.xpath == ["root", "child"]
    
    def test_violation_string_representation(self):
        """Test Violation string representation."""
        violation = Violation(
            rule=RuleId.BASIC_HAS_NO_CHILDREN,
            severity=Severity.ERROR,
            message="Basic state cannot have children",
            xpath=["root", "bad_state"]
        )
        
        str_repr = str(violation)
        assert "ERROR" in str_repr
        assert "BASIC_HAS_NO_CHILDREN" in str_repr
        assert "Basic state cannot have children" in str_repr
        assert "root/bad_state" in str_repr
    
    def test_violation_without_xpath(self):
        """Test Violation without xpath."""
        violation = Violation(
            rule=RuleId.UNIQUE_STATE_LABELS,
            severity=Severity.WARNING,
            message="Test message"
        )
        
        str_repr = str(violation)
        assert "WARNING" in str_repr
        assert "Test message" in str_repr
        assert "root/bad_state" not in str_repr


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_validation_result_valid(self):
        """Test ValidationResult with no violations."""
        result = ValidationResult([])
        
        assert result.is_valid
        assert not result.has_warnings
        assert not result.has_errors
        assert result.get_errors() == []
        assert result.get_warnings() == []
        assert result.get_info() == []
        assert bool(result) == True
        assert str(result) == "Valid"
    
    def test_validation_result_with_warnings(self):
        """Test ValidationResult with warnings only."""
        warning = Violation(
            rule=RuleId.UNIQUE_STATE_LABELS,
            severity=Severity.WARNING,
            message="Warning message"
        )
        
        result = ValidationResult([warning])
        
        assert result.is_valid  # No errors, so still valid
        assert result.has_warnings
        assert not result.has_errors
        assert len(result.get_warnings()) == 1
        assert result.get_errors() == []
        assert bool(result) == True
        assert "Valid with 1 warning(s)" in str(result)
    
    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        error = Violation(
            rule=RuleId.BASIC_HAS_NO_CHILDREN,
            severity=Severity.ERROR,
            message="Error message"
        )
        warning = Violation(
            rule=RuleId.UNIQUE_STATE_LABELS,
            severity=Severity.WARNING,
            message="Warning message"
        )
        
        result = ValidationResult([error, warning])
        
        assert not result.is_valid
        assert result.has_warnings
        assert result.has_errors
        assert len(result.get_errors()) == 1
        assert len(result.get_warnings()) == 1
        assert bool(result) == False
        assert "Invalid: 1 error(s), 1 warning(s)" in str(result)
    
    def test_validation_result_with_info(self):
        """Test ValidationResult with info violations."""
        info = Violation(
            rule=RuleId.UNIQUE_STATE_LABELS,
            severity=Severity.INFO,
            message="Info message"
        )
        
        result = ValidationResult([info])
        
        assert result.is_valid
        assert not result.has_warnings
        assert not result.has_errors
        assert len(result.get_info()) == 1


class TestLocalValidator:
    """Test LocalValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = LocalValidator()
    
    def test_valid_simple_statechart(self):
        """Test validation of a valid simple statechart."""
        self.setUp()
        
        # Create a valid simple statechart
        root = basic_state("Root")
        chart = statechart(root)
        
        result = self.validator.validate_chart(chart)
        
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings
    
    def test_valid_hierarchical_statechart(self):
        """Test validation of a valid hierarchical statechart."""
        self.setUp()
        
        # Create a valid hierarchical statechart
        child1 = basic_state("Child1", is_initial=True)
        child2 = basic_state("Child2")
        parent = normal_state("Parent", [child1, child2])
        
        chart = statechart(parent)
        
        result = self.validator.validate_chart(chart)
        
        assert result.is_valid
        assert not result.has_errors
    
    def test_duplicate_state_labels(self):
        """Test detection of duplicate state labels."""
        self.setUp()
        
        # Create statechart with duplicate labels
        child1 = basic_state("Duplicate")
        child2 = basic_state("Duplicate")  # Same label!
        parent = normal_state("Parent", [child1, child2])
        
        chart = statechart(parent)
        
        result = self.validator.validate_chart(chart)
        
        assert not result.is_valid
        assert result.has_errors
        
        errors = result.get_errors()
        assert len(errors) == 1
        assert errors[0].rule == RuleId.UNIQUE_STATE_LABELS
        assert "Duplicate" in errors[0].message
    
    def test_basic_state_with_children(self):
        """Test detection of basic states with children."""
        self.setUp()
        
        # Create basic state with children (invalid)
        child = basic_state("Child")
        invalid_basic = basic_state("InvalidBasic")
        invalid_basic.children = [child]  # Manually add child to basic state
        
        chart = statechart(invalid_basic)
        
        result = self.validator.validate_chart(chart)
        
        assert not result.is_valid
        assert result.has_errors
        
        errors = result.get_errors()
        assert len(errors) == 1
        assert errors[0].rule == RuleId.BASIC_HAS_NO_CHILDREN
        assert "InvalidBasic" in errors[0].message
    
    def test_compound_state_without_children(self):
        """Test detection of compound states without children."""
        self.setUp()
        
        # Create compound state without children (invalid)
        empty_normal = normal_state("EmptyNormal", [])  # No children
        
        chart = statechart(empty_normal)
        
        result = self.validator.validate_chart(chart)
        
        assert not result.is_valid
        assert result.has_errors
        
        errors = result.get_errors()
        assert len(errors) == 1
        assert errors[0].rule == RuleId.COMPOUND_HAS_CHILDREN
        assert "EmptyNormal" in errors[0].message
    
    def test_normal_state_without_initial_child(self):
        """Test detection of normal states without initial child."""
        self.setUp()
        
        # Create normal state where no child is marked as initial
        child1 = basic_state("Child1")  # Not initial
        child2 = basic_state("Child2")  # Not initial
        parent = normal_state("Parent", [child1, child2])
        
        chart = statechart(parent)
        
        result = self.validator.validate_chart(chart)
        
        assert not result.is_valid
        assert result.has_errors
        
        errors = result.get_errors()
        assert len(errors) == 1
        assert errors[0].rule == RuleId.SINGLE_DEFAULT_CHILD
        assert "exactly one initial child" in errors[0].message
    
    def test_normal_state_with_multiple_initial_children(self):
        """Test detection of normal states with multiple initial children."""
        self.setUp()
        
        # Create normal state where multiple children are marked as initial
        child1 = basic_state("Child1", is_initial=True)  # Initial
        child2 = basic_state("Child2", is_initial=True)  # Also initial!
        parent = normal_state("Parent", [child1, child2])
        
        chart = statechart(parent)
        
        result = self.validator.validate_chart(chart)
        
        assert not result.is_valid
        assert result.has_errors
        
        errors = result.get_errors()
        assert len(errors) == 1
        assert errors[0].rule == RuleId.SINGLE_DEFAULT_CHILD
        assert "multiple initial children" in errors[0].message
    
    def test_multiple_validation_errors(self):
        """Test detection of multiple validation errors."""
        self.setUp()
        
        # Create statechart with multiple problems
        duplicate_child1 = basic_state("Duplicate")
        duplicate_child2 = basic_state("Duplicate")  # Duplicate label
        empty_normal = normal_state("EmptyNormal", [])  # No children
        parent = normal_state("Parent", [duplicate_child1, duplicate_child2, empty_normal])
        
        chart = statechart(parent)
        
        result = self.validator.validate_chart(chart)
        
        assert not result.is_valid
        assert result.has_errors
        
        errors = result.get_errors()
        # Should have at least 2 errors: duplicate labels and empty compound state
        assert len(errors) >= 2
        
        error_rules = [error.rule for error in errors]
        assert RuleId.UNIQUE_STATE_LABELS in error_rules
        assert RuleId.COMPOUND_HAS_CHILDREN in error_rules
    
    def test_valid_parallel_state(self):
        """Test validation of parallel states."""
        self.setUp()
        
        # Create valid parallel statechart
        child1 = basic_state("Child1", is_initial=True)
        child2 = basic_state("Child2", is_initial=True)
        parallel_parent = parallel_state("ParallelParent", [child1, child2])
        
        chart = statechart(parallel_parent)
        
        result = self.validator.validate_chart(chart)
        
        # Parallel states don't require exactly one initial child
        # (all children are conceptually active)
        assert result.is_valid


class TestValidateStatechartFunction:
    """Test the validate_statechart convenience function."""
    
    def test_validate_valid_statechart(self):
        """Test validating a valid statechart."""
        child = basic_state("Child", is_initial=True)
        parent = normal_state("Parent", [child])
        chart = statechart(parent)
        
        result = validate_statechart(chart)
        
        assert result.is_valid
        assert isinstance(result, ValidationResult)
    
    def test_validate_invalid_statechart(self):
        """Test validating an invalid statechart."""
        # Create invalid statechart (basic state with children)
        child = basic_state("Child")
        invalid_basic = basic_state("InvalidBasic")
        invalid_basic.children = [child]
        
        chart = statechart(invalid_basic)
        
        result = validate_statechart(chart)
        
        assert not result.is_valid
        assert result.has_errors


class TestComplexValidationScenarios:
    """Test complex validation scenarios."""
    
    def test_deeply_nested_hierarchy(self):
        """Test validation of deeply nested hierarchy."""
        # Create a deep hierarchy
        leaf = basic_state("Leaf", is_initial=True)
        level3 = normal_state("Level3", [leaf])
        level2 = normal_state("Level2", [level3], is_initial=True)
        level1 = normal_state("Level1", [level2])
        root = normal_state("Root", [level1], is_initial=True)
        
        chart = statechart(root)
        
        result = validate_statechart(chart)
        
        assert result.is_valid
    
    def test_mixed_state_types(self):
        """Test validation with mixed state types."""
        # Create hierarchy with different state types
        basic1 = basic_state("Basic1", is_initial=True)
        basic2 = basic_state("Basic2", is_initial=True)
        
        normal_child = basic_state("NormalChild", is_initial=True)
        normal_parent = normal_state("NormalParent", [normal_child])
        
        parallel_parent = parallel_state("ParallelParent", [basic1, basic2, normal_parent])
        
        chart = statechart(parallel_parent)
        
        result = validate_statechart(chart)
        
        assert result.is_valid
    
    def test_empty_statechart(self):
        """Test validation of statechart with minimal content."""
        root = basic_state("Root")
        chart = statechart(root)
        
        result = validate_statechart(chart)
        
        assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__])