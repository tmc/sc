package semantics

import (
	"fmt"
	"testing"

	"github.com/tmc/sc"
)

func ExampleStatechart_Children() {
	chart := exampleStatechart1
	children, err := chart.Children("On")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("Children of On:", children)
	// Output: Children of On: [Turnstile Control Card Reader Control]
}

func ExampleStatechart_ChildrenStar() {
	chart := exampleStatechart1
	childrenStar, err := chart.ChildrenStar("On")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("ChildrenStar of On:", childrenStar)
	// Output: ChildrenStar of On: [On Turnstile Control Blocked Unblocked Card Reader Control Ready Card Entered Turnstile Unblocked]
}

func ExampleStatechart_AncestrallyRelated() {
	chart := exampleStatechart1
	related, err := chart.AncestrallyRelated("On", "Ready")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("On and Ready ancestrally related:", related)
	// Output: On and Ready ancestrally related: true
}

func ExampleStatechart_LeastCommonAncestor() {
	chart := exampleStatechart1
	lca, err := chart.LeastCommonAncestor("Blocked", "Ready")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("LCA of Blocked and Ready:", lca)
	// Output: LCA of Blocked and Ready: On
}

func ExampleStatechart_Orthogonal() {
	chart := exampleStatechart1
	orthogonal, err := chart.Orthogonal("Blocked", "Ready")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("Blocked and Ready orthogonal:", orthogonal)
	// Output: Blocked and Ready orthogonal: true
}

func ExampleStatechart_Consistent() {
	chart := exampleStatechart1
	consistent, err := chart.Consistent("On", "Blocked", "Ready")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("On, Blocked, and Ready consistent:", consistent)
	// Output: On, Blocked, and Ready consistent: true
}

func ExampleStatechart_DefaultCompletion() {
	chart, err := exampleStatechart1.Normalize()
	if err != nil {
		fmt.Println("Error normalizing chart:", err)
		return
	}
	completion, err := chart.DefaultCompletion("On")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println("Default completion of On:", completion)
	// Output: Default completion of On: [On Turnstile Control Blocked Card Reader Control Ready]
}

func TestExampleStatechart(t *testing.T) {
	// This test ensures that the example statechart is valid
	if err := exampleStatechart1.Validate(); err != nil {
		t.Errorf("Example statechart is invalid: %v", err)
	}
}

// exampleStatechart1 is the example chart from the R. Eshuis paper.
var exampleStatechart1 = NewStatechart(&sc.Statechart{
	RootState: &sc.State{
		Children: []*sc.State{
			{
				Label:     "Off",
				IsInitial: true,
			},
			{
				Label: "On",
				Type:  sc.StateTypeParallel,
				Children: []*sc.State{
					{
						Label: "Turnstile Control",
						Children: []*sc.State{
							{
								Label:     "Blocked",
								IsInitial: true,
							},
							{
								Label: "Unblocked",
							},
						},
					},
					{
						Label: "Card Reader Control",
						Children: []*sc.State{
							{
								Label:     "Ready",
								IsInitial: true,
							},
							{
								Label: "Card Entered",
							},
							{
								Label: "Turnstile Unblocked",
							},
						},
					},
				},
			},
		},
	},
})
