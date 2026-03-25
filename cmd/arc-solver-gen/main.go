package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
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
	outDir := flag.String("out", "ml/data/solver", "Output directory")
	count := flag.Int("count", 2000, "Number of examples to generate")
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0755); err != nil {
		panic(err)
	}

	outFile, err := os.Create(filepath.Join(*outDir, "recolor_data.json"))
	if err != nil {
		panic(err)
	}
	defer outFile.Close()

	rand.Seed(time.Now().UnixNano())

	// Split 90/10
	trainCount := int(float64(*count) * 0.9)
	testCount := *count - trainCount

	data := ArcData{
		Train: generateMixedPairs(trainCount),
		Test:  generateMixedPairs(testCount),
	}

	encoder := json.NewEncoder(outFile)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(data); err != nil {
		panic(err)
	}

	fmt.Printf("Generated %d pairs to %s\n", *count, outFile.Name())
}

func generateMixedPairs(n int) []Pair {
	pairs := make([]Pair, n)
	for i := 0; i < n; i++ {
		if rand.Float32() < 0.5 {
			pairs[i] = generateRecolorExample()
		} else {
			pairs[i] = generateGravityExample()
		}
	}
	return pairs
}

func generateRecolorExample() Pair {
	// 1. Generate random grid
	rows := rand.Intn(5) + 2 // 2-6
	cols := rand.Intn(5) + 2 // 2-6

	grid, colorsUsed := generateGrid(rows, cols)

	// 2. Select source and target colors
	// Ensure source exists
	sourceColor := colorsUsed[rand.Intn(len(colorsUsed))]
	targetColor := rand.Intn(10)
	for targetColor == sourceColor {
		targetColor = rand.Intn(10)
	}

	// 3. Format Input
	gridStr := gridToString(grid)
	inputStr := fmt.Sprintf("Grid:\n%s\nTask: Replace %d with %d.", gridStr, sourceColor, targetColor)

	// 4. Generate Output with GPro Guidance
	// Goal, Reasoning, Plan, Code

	outputStr := fmt.Sprintf(`Goal: Replace all instances of color %d with color %d in the grid.
Reasoning: The task asks to modify pixels of a specific color (%d) to a new color (%d) while keeping other pixels unchanged. A simple iteration over the grid is sufficient.
Plan:
1. Create a copy of the grid to avoid modifying the input in place.
2. Iterate through each row and column.
3. If a cell has value %d, change it to %d.
4. Return the new grid.
Code:
def solve(grid):
    new_grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if new_grid[r][c] == %d:
                new_grid[r][c] = %d
    return new_grid`, sourceColor, targetColor, sourceColor, targetColor, sourceColor, targetColor, sourceColor, targetColor)

	return Pair{
		Input:  inputStr,
		Output: outputStr,
	}
}

func generateGrid(rows, cols int) ([][]int, []int) {
	grid := make([][]int, rows)
	unique := make(map[int]bool)

	for r := 0; r < rows; r++ {
		grid[r] = make([]int, cols)
		for c := 0; c < cols; c++ {
			val := rand.Intn(10)
			grid[r][c] = val
			unique[val] = true
		}
	}

	colors := []int{}
	for k := range unique {
		colors = append(colors, k)
	}
	return grid, colors
}

func generateGravityExample() Pair {
	// 1. Generate random grid with sparse pixels
	rows := rand.Intn(5) + 3 // 3-7
	cols := rand.Intn(5) + 3 // 3-7

	grid, _ := generateGrid(rows, cols)

	// Sparsify: make 70% of pixels 0 (black)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if rand.Float32() < 0.7 {
				grid[r][c] = 0
			}
		}
	}

	// 2. Format Input
	gridStr := gridToString(grid)
	inputStr := fmt.Sprintf("Grid:\n%s\nTask: Move all non-black pixels to the bottom of the grid.", gridStr)

	// 3. Generate Output with GEPA
	outputStr := fmt.Sprintf(`Goal: Move all non-black pixels to the bottom of the grid (Gravity).
Evidence: The grid is %dx%d. Some pixels have non-zero values.
Plan:
1. Create a new grid of zeros with dimensions %dx%d.
2. Iterate through each column from 0 to %d.
3. For each column, collect all non-zero values from the input grid.
4. Place these collected values at the bottom of the new grid's column, filling upwards.
5. Return the new grid.
Action:
def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for c in range(cols):
        # Collect non-zero values
        values = []
        for r in range(rows):
            if grid[r][c] != 0:
                values.append(grid[r][c])
        
        # Place at bottom
        # Bottom is rows-1. We fill from bottom up.
        # values index -1 goes to rows-1
        # values index -2 goes to rows-2 ...
        for i in range(len(values)):
            val = values[len(values) - 1 - i]
            new_grid[rows - 1 - i][c] = val
            
    return new_grid`,
		rows, cols,
		rows, cols,
		cols-1)

	return Pair{
		Input:  inputStr,
		Output: outputStr,
	}
}

func gridToString(grid [][]int) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, row := range grid {
		rowJson, _ := json.Marshal(row)
		sb.WriteString(string(rowJson))
		if i < len(grid)-1 {
			sb.WriteString(", ")
		}
	}
	sb.WriteString("]")
	return sb.String()
}
