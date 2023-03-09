package semantics

import "github.com/tmc/sc"

// This is the example chart from the R. Eshuis paper.
var exampleStatechart1 = &Statechart{&sc.Statechart{
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
					},
					{
						Label: "Card Reader Control",
						Children: []*sc.State{
							{
								Label:     "Ready",
								IsInitial: true,
							},
						},
					},
				},
			},
		},
	},
}}
