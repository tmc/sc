package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

type Pair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}

type ArcData struct {
	Train []Pair `json:"train"`
	Test  []Pair `json:"test"`
}

func main() {
	outDir := flag.String("out", "ml/data", "Output directory")
	count := flag.Int("count", 1000, "Number of examples to generate")
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0755); err != nil {
		panic(err)
	}

	outFile, err := os.Create(filepath.Join(*outDir, "arc_data.json"))
	if err != nil {
		panic(err)
	}
	defer outFile.Close()

	// Split 90/10
	trainCount := int(float64(*count) * 0.9)
	testCount := *count - trainCount

	data := ArcData{
		Train: generatePairs(trainCount),
		Test:  generatePairs(testCount),
	}

	encoder := json.NewEncoder(outFile)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(data); err != nil {
		panic(err)
	}
}

func generatePairs(n int) []Pair {
	pairs := make([]Pair, n)
	for i := 0; i < n; i++ {
		pairs[i] = generateEntry()
	}
	return pairs
}

func generateEntry() Pair {
	kind := rand.Intn(3)
	var prompt, code string

	switch kind {
	case 0:
		prompt, code = generateLinear()
	case 1:
		prompt, code = generateCycle()
	case 2:
		prompt, code = generateTrafficLight()
	}

	return Pair{
		Input:  prompt,
		Output: code,
	}
}

func generateLinear() (string, string) {
	n := rand.Intn(4) + 2 // 2 to 5 states
	states := make([]string, n)
	for i := 0; i < n; i++ {
		states[i] = fmt.Sprintf("s%d", i+1)
	}

	prompt := fmt.Sprintf("Create a statechart with %d states that transitions linearly from %s to %s.", n, states[0], states[n-1])

	var sb strings.Builder
	sb.WriteString("load(\"sc\", \"sc\")\n\n")
	sb.WriteString("def machine():\n")
	sb.WriteString("    return sc.machine(\n")
	sb.WriteString("        name = \"linear\",\n")
	sb.WriteString("        states = [\n")

	for i, s := range states {
		initial := ""
		if i == 0 {
			initial = ", initial=True"
		}

		trans := ""
		if i < n-1 {
			trans = fmt.Sprintf(", transitions=[\n                sc.transition(to=\"%s\", event=\"NEXT\"),\n            ]", states[i+1])
		}

		sb.WriteString(fmt.Sprintf("            sc.state(\"%s\"%s%s),\n", s, initial, trans))
	}
	sb.WriteString("        ],\n")
	sb.WriteString("    )\n")

	return prompt, sb.String()
}

func generateCycle() (string, string) {
	n := rand.Intn(3) + 2 // 2 to 4 states
	states := make([]string, n)
	labels := []string{"A", "B", "C", "D", "E"}
	for i := 0; i < n; i++ {
		states[i] = labels[i]
	}

	prompt := fmt.Sprintf("Create a cyclical statechart with states %s.", strings.Join(states, ", "))

	var sb strings.Builder
	sb.WriteString("load(\"sc\", \"sc\")\n\n")
	sb.WriteString("def machine():\n")
	sb.WriteString("    return sc.machine(\n")
	sb.WriteString("        name = \"cycle\",\n")
	sb.WriteString("        states = [\n")

	for i, s := range states {
		initial := ""
		if i == 0 {
			initial = ", initial=True"
		}

		next := states[(i+1)%n]
		trans := fmt.Sprintf(", transitions=[\n                sc.transition(to=\"%s\", event=\"STEP\"),\n            ]", next)

		sb.WriteString(fmt.Sprintf("            sc.state(\"%s\"%s%s),\n", s, initial, trans))
	}
	sb.WriteString("        ],\n")
	sb.WriteString("    )\n")

	return prompt, sb.String()
}

func generateTrafficLight() (string, string) {
	prompt := "Create a standard traffic light statechart."

	code := `load("sc", "sc")

def machine():
    return sc.machine(
        name = "traffic_light",
        states = [
            sc.state("green", initial=True, transitions=[
                sc.transition(to="yellow", event="TIMER"),
            ]),
            sc.state("yellow", transitions=[
                sc.transition(to="red", event="TIMER"),
            ]),
            sc.state("red", transitions=[
                sc.transition(to="green", event="TIMER"),
            ]),
        ],
    )`
	return prompt, code
}
