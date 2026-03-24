package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: stately-scraper <command> [args]")
		fmt.Println("Commands: projects, discover, asset, stats, cleanup, enrich, convert")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "projects":
		runProjects(os.Args[2:])
	case "discover":
		runDiscover(os.Args[2:])
	case "asset":
		runAsset(os.Args[2:])
	case "stats":
		runStats(os.Args[2:])
	case "cleanup":
		runCleanup(os.Args[2:])
	case "enrich":
		runEnrich(os.Args[2:])
	case "convert":
		runConvert(os.Args[2:])
	default:
		log.Fatalf("Unknown command: %s", os.Args[1])
	}
}

// Progress tracks scraping statistics
type Progress struct {
	downloaded atomic.Int64
	skipped    atomic.Int64
	errors     atomic.Int64
	invalid    atomic.Int64
	startTime  time.Time
	mu         sync.Mutex
	lastErrors []string // recent errors for reporting
	totalPages int      // estimated total pages (0 if unknown)
	pagesCount atomic.Int64
	pageTimes  []time.Duration // track last N page durations for ETA
}

func NewProgress() *Progress {
	return &Progress{
		startTime:  time.Now(),
		lastErrors: make([]string, 0, 10),
		pageTimes:  make([]time.Duration, 0, 20),
	}
}

func (p *Progress) SetTotalPages(total int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.totalPages = total
}

func (p *Progress) AddPageTime(d time.Duration) {
	p.pagesCount.Add(1)
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pageTimes = append(p.pageTimes, d)
	// Keep only last 20 page times for moving average
	if len(p.pageTimes) > 20 {
		p.pageTimes = p.pageTimes[1:]
	}
}

func (p *Progress) AvgPageTime() time.Duration {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.pageTimes) == 0 {
		return 0
	}
	var total time.Duration
	for _, d := range p.pageTimes {
		total += d
	}
	return total / time.Duration(len(p.pageTimes))
}

func (p *Progress) ETA(currentPage int) string {
	p.mu.Lock()
	totalPages := p.totalPages
	p.mu.Unlock()

	if totalPages == 0 {
		return "unknown"
	}

	remaining := totalPages - currentPage
	if remaining <= 0 {
		return "done"
	}

	avgTime := p.AvgPageTime()
	if avgTime == 0 {
		return "calculating..."
	}

	eta := time.Duration(remaining) * avgTime
	if eta < time.Minute {
		return fmt.Sprintf("%ds", int(eta.Seconds()))
	} else if eta < time.Hour {
		return fmt.Sprintf("%dm", int(eta.Minutes()))
	}
	return fmt.Sprintf("%dh%dm", int(eta.Hours()), int(eta.Minutes())%60)
}

func (p *Progress) AddDownloaded() { p.downloaded.Add(1) }
func (p *Progress) AddSkipped()    { p.skipped.Add(1) }
func (p *Progress) AddInvalid()    { p.invalid.Add(1) }

func (p *Progress) AddError(err string) {
	p.errors.Add(1)
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.lastErrors) >= 10 {
		p.lastErrors = p.lastErrors[1:]
	}
	p.lastErrors = append(p.lastErrors, err)
}

func (p *Progress) Stats() (downloaded, skipped, errors, invalid int64) {
	return p.downloaded.Load(), p.skipped.Load(), p.errors.Load(), p.invalid.Load()
}

func (p *Progress) Rate() float64 {
	total := p.downloaded.Load() + p.skipped.Load()
	elapsed := time.Since(p.startTime).Seconds()
	if elapsed == 0 {
		return 0
	}
	return float64(total) / elapsed
}

func (p *Progress) Summary() string {
	d, s, e, i := p.Stats()
	rate := p.Rate()
	elapsed := time.Since(p.startTime).Round(time.Second)
	return fmt.Sprintf("downloaded=%d skipped=%d errors=%d invalid=%d | %.1f/sec | %v elapsed",
		d, s, e, i, rate, elapsed)
}

// State persists scraper progress for resume
type State struct {
	LastPage      int       `json:"last_page"`
	TotalPages    int       `json:"total_pages,omitempty"`
	TotalDownload int64     `json:"total_downloaded"`
	TotalSkipped  int64     `json:"total_skipped"`
	TotalErrors   int64     `json:"total_errors"`
	LastRun       time.Time `json:"last_run"`
	LastErrors    []string  `json:"last_errors,omitempty"`
	ErrorPages    []int     `json:"error_pages,omitempty"` // pages with errors for retry
}

func loadState(filename string) (*State, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return &State{LastPage: 1}, nil
	}
	var state State
	if err := json.Unmarshal(data, &state); err != nil {
		return &State{LastPage: 1}, nil
	}
	return &state, nil
}

func saveState(filename string, state *State) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func runProjects(args []string) {
	fs := flag.NewFlagSet("projects", flag.ExitOnError)
	cookie := fs.String("cookie", os.Getenv("STATELY_COOKIE"), "Cookie header value (or STATELY_COOKIE env var)")
	limit := fs.Int("limit", 50, "Limit for pagination")
	offset := fs.Int("offset", 0, "Offset for pagination")
	teams := fs.Bool("teams", true, "Include team projects")
	personal := fs.Bool("personal", true, "Include personal projects")
	sleep := fs.Duration("sleep", 0, "Sleep duration before request (e.g. 500ms)")
	fs.Parse(args)

	if *cookie == "" {
		log.Fatal("Cookie is required")
	}

	client := NewClient(*cookie, *sleep)
	baseURL := "https://stately.ai/registry/api/trpc/projects.many,projects.many"

	// Construct input JSON
	input := make(map[string]interface{})
	idx := 0
	if *personal {
		input[fmt.Sprintf("%d", idx)] = map[string]interface{}{
			"json": map[string]interface{}{
				"limit":          *limit,
				"offset":         *offset,
				"source":         "editor",
				"myProjectsOnly": true,
			},
		}
		idx++
	}
	if *teams {
		input[fmt.Sprintf("%d", idx)] = map[string]interface{}{
			"json": map[string]interface{}{
				"limit":              *limit,
				"offset":             *offset,
				"source":             "editor",
				"myTeamProjectsOnly": true,
			},
		}
		idx++
	}

	inputJSON, err := json.Marshal(input)
	if err != nil {
		log.Fatalf("Failed to marshal input: %v", err)
	}

	params := url.Values{}
	params.Add("batch", "1")
	params.Add("input", string(inputJSON))

	reqURL := fmt.Sprintf("%s?%s", baseURL, params.Encode())

	req, err := http.NewRequest("GET", reqURL, nil)
	if err != nil {
		log.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("content-type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		log.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Failed to read body: %v", err)
	}

	fmt.Println(string(body))
}

func runDiscover(args []string) {
	fs := flag.NewFlagSet("discover", flag.ExitOnError)
	cookie := fs.String("cookie", os.Getenv("STATELY_COOKIE"), "Cookie header value (or STATELY_COOKIE env var)")
	page := fs.Int("page", 1, "Page number to start from")
	query := fs.String("query", "", "Search query")
	limit := fs.Int("limit", 50, "Results per page")
	sleep := fs.Duration("sleep", 500*time.Millisecond, "Sleep duration between requests")
	all := fs.Bool("all", false, "Scrape all pages starting from -page")
	output := fs.String("output", "charts", "Output directory for downloaded charts")
	download := fs.Bool("download", false, "Download project definitions")
	full := fs.Bool("full", false, "Fetch full machine definitions via editorData API")
	resume := fs.Bool("resume", false, "Resume from last saved state")
	workers := fs.Int("workers", 4, "Number of concurrent download workers")
	force := fs.Bool("force", false, "Re-download even if file exists")
	verbose := fs.Bool("v", false, "Verbose output (show each file)")
	fs.Parse(args)

	stateFile := "scraper_state.json"
	progress := NewProgress()

	// Load or initialize state
	state, _ := loadState(stateFile)
	startPage := *page
	if *resume && state.LastPage > 1 {
		startPage = state.LastPage
		if state.TotalPages > 0 {
			progress.SetTotalPages(state.TotalPages)
		}
		// Initialize progress counters from saved state
		progress.downloaded.Store(state.TotalDownload)
		progress.skipped.Store(state.TotalSkipped)
		progress.errors.Store(state.TotalErrors)
		fmt.Fprintf(os.Stderr, "Resuming from page %d/%d (previous: downloaded=%d skipped=%d errors=%d, error_pages=%d)\n",
			startPage, state.TotalPages, state.TotalDownload, state.TotalSkipped, state.TotalErrors, len(state.ErrorPages))
	}

	client := NewClient(*cookie, *sleep)
	baseURL := "https://stately.ai/registry/api/trpc/search.getDiscoverPageResults"

	currentPage := startPage
	pagesProcessed := 0

	for {
		pageStart := time.Now()

		// Construct input JSON - filters must be {} not null
		input := map[string]interface{}{
			"0": map[string]interface{}{
				"json": map[string]interface{}{
					"query":          *query,
					"filters":        map[string]interface{}{}, // Must be object, not null
					"page":           currentPage,
					"resultsPerPage": *limit,
				},
			},
		}

		inputJSON, err := json.Marshal(input)
		if err != nil {
			log.Fatalf("\nFailed to marshal input: %v", err)
		}

		params := url.Values{}
		params.Add("batch", "1")
		params.Add("input", string(inputJSON))

		reqURL := fmt.Sprintf("%s?%s", baseURL, params.Encode())

		req, err := http.NewRequest("GET", reqURL, nil)
		if err != nil {
			log.Fatalf("\nFailed to create request: %v", err)
		}
		req.Header.Set("content-type", "application/json")

		var resp *http.Response
		var body []byte
		maxRetries := 5
		for i := 0; i < maxRetries; i++ {
			resp, err = client.Do(req)
			if err == nil {
				body, err = io.ReadAll(resp.Body)
				resp.Body.Close()
				if err == nil {
					break
				}
			}
			fmt.Fprintf(os.Stderr, "\n  [Retry %d/%d for page %d: %v]\n", i+1, maxRetries, currentPage, err)
			time.Sleep(2 * time.Second * time.Duration(i+1))
			if i == maxRetries-1 {
				log.Fatalf("\nRequest failed after %d retries: %v", maxRetries, err)
			}
		}

		// Parse the TRPC response
		var response struct {
			Result struct {
				Data struct {
					Json struct {
						Results []struct {
							ProjectID              string `json:"project_id"`
							ResultName             string `json:"result_name"`
							ResultID               string `json:"result_id"`
							OwnerName              string `json:"owner_name"`
							OwnerID                string `json:"owner_id"`
							ResultProjectVersionID string `json:"result_project_version_id"`
							Visibility             string `json:"visibility"`
							MachineSource          string `json:"machine_source"`
						} `json:"results"`
						NumberOfResults int `json:"numberOfResults"`
						NumberOfPages   int `json:"numberOfPages"`
					} `json:"json"`
				} `json:"data"`
			} `json:"result"`
		}

		var batchResponse []interface{}
		if err := json.Unmarshal(body, &batchResponse); err != nil {
			log.Printf("\nFailed to unmarshal batch response: %v", err)
		} else if len(batchResponse) > 0 {
			if firstItemBytes, err := json.Marshal(batchResponse[0]); err == nil {
				json.Unmarshal(firstItemBytes, &response)
			}
		}

		results := response.Result.Data.Json.Results
		totalResults := response.Result.Data.Json.NumberOfResults
		totalPages := response.Result.Data.Json.NumberOfPages

		// Set total pages on first response
		if totalPages > 0 && progress.totalPages == 0 {
			progress.SetTotalPages(totalPages)
			state.TotalPages = totalPages
			fmt.Fprintf(os.Stderr, "Total: %d results across %d pages\n", totalResults, totalPages)
		}

		if len(results) == 0 {
			fmt.Fprintf(os.Stderr, "\rPage %d: empty - finished!                              \n", currentPage)
			break
		}

		if *download {
			var wg sync.WaitGroup
			sem := make(chan struct{}, *workers)
			pageDownloaded := atomic.Int64{}
			pageSkipped := atomic.Int64{}
			pageErrors := atomic.Int64{}

			for _, r := range results {
				if r.ProjectID == "" {
					continue
				}
				wg.Add(1)
				go func(r struct {
					ProjectID              string `json:"project_id"`
					ResultName             string `json:"result_name"`
					ResultID               string `json:"result_id"`
					OwnerName              string `json:"owner_name"`
					OwnerID                string `json:"owner_id"`
					ResultProjectVersionID string `json:"result_project_version_id"`
					Visibility             string `json:"visibility"`
					MachineSource          string `json:"machine_source"`
				}) {
					defer wg.Done()
					sem <- struct{}{}
					defer func() { <-sem }()

					// Use IDs for directory structure: owner_id/project_id/result_id.json
					ownerID := r.OwnerID
					if ownerID == "" {
						ownerID = "unknown"
					}
					projectID := r.ProjectID
					if projectID == "" {
						progress.AddError("empty project_id")
						pageErrors.Add(1)
						return
					}
					resultID := r.ResultID
					if resultID == "" {
						resultID = "default"
					}

					projectDir := fmt.Sprintf("%s/%s/%s", *output, ownerID, projectID)
					if err := os.MkdirAll(projectDir, 0755); err != nil {
						progress.AddError(fmt.Sprintf("mkdir %s: %v", projectDir, err))
						pageErrors.Add(1)
						return
					}

					filename := fmt.Sprintf("%s/%s.json", projectDir, resultID)

					// Skip if exists (unless force)
					if !*force {
						if _, err := os.Stat(filename); err == nil {
							progress.AddSkipped()
							pageSkipped.Add(1)
							if *verbose {
								fmt.Fprintf(os.Stderr, "  skip: %s\n", filename)
							}
							return
						}
					}

					// Fetch and save data
					var projectData []byte
					if *full && r.ResultProjectVersionID != "" && r.ResultID != "" {
						// Fetch full machine definition via editorData API
						data, err := fetchEditorData(client, r.ResultProjectVersionID, r.ResultID)
						if err != nil {
							// Fall back to discovery metadata on error
							if *verbose {
								fmt.Fprintf(os.Stderr, "  editorData error for %s: %v, saving metadata\n", r.ProjectID, err)
							}
							projectData, _ = json.MarshalIndent(r, "", "  ")
						} else {
							projectData = data
						}
					} else {
						// Save discovery metadata only
						projectData, _ = json.MarshalIndent(r, "", "  ")
					}
					if projectData == nil {
						progress.AddError(fmt.Sprintf("no data for %s", r.ProjectID))
						pageErrors.Add(1)
						return
					}

					// Atomic write
					tmpFilename := filename + ".tmp"
					if err := os.WriteFile(tmpFilename, projectData, 0644); err != nil {
						progress.AddError(fmt.Sprintf("write %s: %v", filename, err))
						pageErrors.Add(1)
						return
					}
					if err := os.Rename(tmpFilename, filename); err != nil {
						progress.AddError(fmt.Sprintf("rename %s: %v", filename, err))
						pageErrors.Add(1)
						os.Remove(tmpFilename)
						return
					}

					progress.AddDownloaded()
					pageDownloaded.Add(1)
					if *verbose {
						fmt.Fprintf(os.Stderr, "  new:  %s\n", filename)
					}
				}(r)
			}
			wg.Wait()

			// Track page timing
			pageDuration := time.Since(pageStart)
			progress.AddPageTime(pageDuration)

			// Track pages with errors for retry
			if pageErrors.Load() > 0 {
				state.ErrorPages = appendUnique(state.ErrorPages, currentPage)
			}

			// Print page summary with ETA
			eta := progress.ETA(currentPage)
			if progress.totalPages > 0 {
				fmt.Fprintf(os.Stderr, "\rPage %d/%d: +%d new, %d skip, %d err (%v) | %s | ETA: %s\n",
					currentPage,
					progress.totalPages,
					pageDownloaded.Load(),
					pageSkipped.Load(),
					pageErrors.Load(),
					pageDuration.Round(time.Millisecond),
					progress.Summary(),
					eta)
			} else {
				fmt.Fprintf(os.Stderr, "\rPage %d: +%d new, %d skip, %d err (%v) | %s\n",
					currentPage,
					pageDownloaded.Load(),
					pageSkipped.Load(),
					pageErrors.Load(),
					pageDuration.Round(time.Millisecond),
					progress.Summary())
			}
		} else {
			// If not downloading, just print raw body to stdout
			fmt.Println(string(body))
		}

		// Save state after each page
		d, s, e, _ := progress.Stats()
		state.LastPage = currentPage + 1
		state.TotalDownload = d
		state.TotalSkipped = s
		state.TotalErrors = e
		state.LastRun = time.Now()
		if err := saveState(stateFile, state); err != nil {
			fmt.Fprintf(os.Stderr, "  [warning: failed to save state: %v]\n", err)
		}

		if !*all {
			break
		}

		pagesProcessed++
		currentPage++
	}

	// Final summary
	fmt.Fprintf(os.Stderr, "\nComplete: %s\n", progress.Summary())
}

func fetchProject(client *http.Client, projectID string) ([]byte, error) {
	baseURL := "https://stately.ai/registry/api/trpc/projects.get"
	input := map[string]interface{}{
		"0": map[string]interface{}{
			"json": map[string]interface{}{
				"projectId": projectID,
			},
		},
	}

	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}

	params := url.Values{}
	params.Add("batch", "1")
	params.Add("input", string(inputJSON))

	reqURL := fmt.Sprintf("%s?%s", baseURL, params.Encode())

	req, err := http.NewRequest("GET", reqURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("content-type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	return io.ReadAll(resp.Body)
}

func sanitizeFilename(name string) string {
	return url.PathEscape(name)
}

// fetchEditorData fetches the full machine definition from the editorData API
func fetchEditorData(client *http.Client, projectVersionID, machineID string) ([]byte, error) {
	baseURL := "https://stately.ai/registry/api/trpc/projects.editorData"

	input := map[string]interface{}{
		"0": map[string]interface{}{
			"json": map[string]interface{}{
				"projectOrMachineId": projectVersionID,
				"selectedMachineId":  machineID,
			},
		},
	}

	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("marshal input: %w", err)
	}

	params := url.Values{}
	params.Add("batch", "1")
	params.Add("input", string(inputJSON))

	reqURL := fmt.Sprintf("%s?%s", baseURL, params.Encode())

	req, err := http.NewRequest("GET", reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("Referer", "https://stately.ai/registry/discover?page=1")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
	}

	// Check for error response
	if isErrorResponse(body) {
		return nil, fmt.Errorf("API error: %s", string(body))
	}

	return body, nil
}

func runAsset(args []string) {
	fs := flag.NewFlagSet("asset", flag.ExitOnError)
	cookie := fs.String("cookie", os.Getenv("STATELY_COOKIE"), "Cookie header value (or STATELY_COOKIE env var)")
	assetURL := fs.String("url", "", "Full URL of the asset to fetch")
	sleep := fs.Duration("sleep", 0, "Sleep duration before request (e.g. 500ms)")
	fs.Parse(args)

	if *assetURL == "" {
		log.Fatal("Asset URL is required")
	}

	client := NewClient(*cookie, *sleep)
	req, err := http.NewRequest("GET", *assetURL, nil)
	if err != nil {
		log.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("Referer", "https://stately.ai/registry/discover?page=1")

	resp, err := client.Do(req)
	if err != nil {
		log.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Failed to read body: %v", err)
	}

	fmt.Println(string(body))
}

// runStats shows statistics about the charts directory
func runStats(args []string) {
	fs := flag.NewFlagSet("stats", flag.ExitOnError)
	dir := fs.String("dir", "charts", "Directory to analyze")
	verify := fs.Bool("verify", false, "Verify JSON validity of all files")
	fs.Parse(args)

	var totalFiles, totalDirs, invalidJSON int64
	var totalBytes int64

	err := walkDir(*dir, func(path string, info os.FileInfo) error {
		if info.IsDir() {
			totalDirs++
			return nil
		}
		totalFiles++
		totalBytes += info.Size()

		if *verify && hasJSONExtension(path) {
			data, err := os.ReadFile(path)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  read error: %s\n", path)
				invalidJSON++
				return nil
			}
			if !json.Valid(data) {
				fmt.Fprintf(os.Stderr, "  invalid JSON: %s\n", path)
				invalidJSON++
			}
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Failed to walk directory: %v", err)
	}

	fmt.Printf("Directory: %s\n", *dir)
	fmt.Printf("Files:     %d\n", totalFiles)
	fmt.Printf("Dirs:      %d\n", totalDirs)
	fmt.Printf("Size:      %.2f MB\n", float64(totalBytes)/(1024*1024))
	if *verify {
		fmt.Printf("Invalid:   %d\n", invalidJSON)
	}

	// Show state file info
	state, err := loadState("scraper_state.json")
	if err == nil && state.LastPage > 1 {
		fmt.Printf("\nScraper state:\n")
		fmt.Printf("  Last page:   %d\n", state.LastPage)
		if state.TotalPages > 0 {
			fmt.Printf("  Total pages: %d\n", state.TotalPages)
			fmt.Printf("  Progress:    %.1f%%\n", float64(state.LastPage)/float64(state.TotalPages)*100)
		}
		fmt.Printf("  Downloaded:  %d\n", state.TotalDownload)
		fmt.Printf("  Skipped:     %d\n", state.TotalSkipped)
		fmt.Printf("  Errors:      %d\n", state.TotalErrors)
		if len(state.ErrorPages) > 0 {
			fmt.Printf("  Error pages: %d (use -retry-errors to retry)\n", len(state.ErrorPages))
			if len(state.ErrorPages) <= 20 {
				fmt.Printf("               %v\n", state.ErrorPages)
			} else {
				fmt.Printf("               %v ... and %d more\n", state.ErrorPages[:20], len(state.ErrorPages)-20)
			}
		}
		if !state.LastRun.IsZero() {
			fmt.Printf("  Last run:    %s\n", state.LastRun.Format(time.RFC3339))
		}
	}
}

// runCleanup removes error response files from the charts directory
// runEnrich reads metadata files and fetches full machine definitions
func runEnrich(args []string) {
	fs := flag.NewFlagSet("enrich", flag.ExitOnError)
	cookie := fs.String("cookie", os.Getenv("STATELY_COOKIE"), "Cookie header value (or STATELY_COOKIE env var)")
	inputDir := fs.String("input", "charts", "Input directory with metadata files")
	outputDir := fs.String("output", "", "Output directory (defaults to input dir, overwrites)")
	batchSize := fs.Int("batch", 10, "Number of machines per API request")
	workers := fs.Int("workers", 4, "Number of concurrent workers")
	sleep := fs.Duration("sleep", 200*time.Millisecond, "Sleep between batches")
	force := fs.Bool("force", false, "Re-fetch even if file looks enriched")
	verbose := fs.Bool("v", false, "Verbose output")
	fs.Parse(args)

	if *outputDir == "" {
		*outputDir = *inputDir
	}

	// Collect all metadata files that need enrichment
	var files []metaFile

	err := walkDir(*inputDir, func(path string, info os.FileInfo) error {
		if info.IsDir() || !hasJSONExtension(path) {
			return nil
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		// Check if already enriched (has "result" wrapper from editorData)
		if !*force && (len(data) > 10 && string(data[:10]) == "[{\"result\"") {
			return nil
		}

		// Parse metadata
		var meta struct {
			ProjectID              string `json:"project_id"`
			ResultID               string `json:"result_id"`
			OwnerID                string `json:"owner_id"`
			ResultProjectVersionID string `json:"result_project_version_id"`
		}
		if err := json.Unmarshal(data, &meta); err != nil {
			return nil
		}

		if meta.ResultProjectVersionID == "" || meta.ResultID == "" {
			return nil
		}

		files = append(files, metaFile{
			path:      path,
			ownerID:   meta.OwnerID,
			projectID: meta.ProjectID,
			versionID: meta.ResultProjectVersionID,
			machineID: meta.ResultID,
		})
		return nil
	})
	if err != nil {
		log.Fatalf("Failed to walk directory: %v", err)
	}

	fmt.Fprintf(os.Stderr, "Found %d files to enrich\n", len(files))
	if len(files) == 0 {
		return
	}

	client := NewClient(*cookie, *sleep)
	progress := NewProgress()

	// Create batches
	var batches [][]metaFile
	for i := 0; i < len(files); i += *batchSize {
		end := i + *batchSize
		if end > len(files) {
			end = len(files)
		}
		batches = append(batches, files[i:end])
	}

	fmt.Fprintf(os.Stderr, "Processing %d batches with %d workers\n", len(batches), *workers)

	// Process batches with workers
	batchCh := make(chan []metaFile, len(batches))
	for _, batch := range batches {
		batchCh <- batch
	}
	close(batchCh)

	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for batch := range batchCh {
				results, err := fetchEditorDataBatch(client, batch)
				if err != nil {
					progress.AddError(fmt.Sprintf("batch fetch: %v", err))
					continue
				}

				// Save each result
				for i, mf := range batch {
					if i >= len(results) {
						progress.AddError(fmt.Sprintf("missing result for %s", mf.path))
						continue
					}

					// Determine output path
					outPath := mf.path
					if *outputDir != *inputDir {
						// Replace input dir prefix with output dir
						relPath := mf.path[len(*inputDir):]
						outPath = *outputDir + relPath
						// Ensure directory exists
						outDir := outPath[:len(outPath)-len("/"+mf.machineID+".json")]
						if err := os.MkdirAll(outDir, 0755); err != nil {
							progress.AddError(fmt.Sprintf("mkdir %s: %v", outDir, err))
							continue
						}
					}

					// Write result
					tmpPath := outPath + ".tmp"
					if err := os.WriteFile(tmpPath, results[i], 0644); err != nil {
						progress.AddError(fmt.Sprintf("write %s: %v", outPath, err))
						continue
					}
					if err := os.Rename(tmpPath, outPath); err != nil {
						progress.AddError(fmt.Sprintf("rename %s: %v", outPath, err))
						os.Remove(tmpPath)
						continue
					}

					progress.AddDownloaded()
					if *verbose {
						fmt.Fprintf(os.Stderr, "  enriched: %s\n", outPath)
					}
				}

				// Progress update
				fmt.Fprintf(os.Stderr, "\r%s", progress.Summary())
			}
		}()
	}

	wg.Wait()
	fmt.Fprintf(os.Stderr, "\nComplete: %s\n", progress.Summary())
}

// fetchEditorDataBatch fetches multiple machine definitions in a single batched API call
func fetchEditorDataBatch(client *http.Client, files []metaFile) ([][]byte, error) {
	if len(files) == 0 {
		return nil, nil
	}

	// Build batched URL with multiple procedure names
	procs := make([]string, len(files))
	for i := range files {
		procs[i] = "projects.editorData"
	}
	baseURL := "https://stately.ai/registry/api/trpc/" + strings.Join(procs, ",")

	// Build input for each
	input := make(map[string]interface{})
	for i, mf := range files {
		input[fmt.Sprintf("%d", i)] = map[string]interface{}{
			"json": map[string]interface{}{
				"projectOrMachineId": mf.versionID,
				"selectedMachineId":  mf.machineID,
			},
		}
	}

	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("marshal input: %w", err)
	}

	params := url.Values{}
	params.Add("batch", "1")
	params.Add("input", string(inputJSON))

	reqURL := fmt.Sprintf("%s?%s", baseURL, params.Encode())

	req, err := http.NewRequest("GET", reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("Referer", "https://stately.ai/registry/discover?page=1")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}

	// Parse the batch response array
	var batchResp []json.RawMessage
	if err := json.Unmarshal(body, &batchResp); err != nil {
		return nil, fmt.Errorf("unmarshal batch: %w", err)
	}

	// Convert each to bytes
	results := make([][]byte, len(batchResp))
	for i, raw := range batchResp {
		// Wrap single result in array to match single-fetch format
		results[i] = []byte("[" + string(raw) + "]")
	}

	return results, nil
}

type metaFile struct {
	path      string
	ownerID   string
	projectID string
	versionID string
	machineID string
}

func runCleanup(args []string) {
	fs := flag.NewFlagSet("cleanup", flag.ExitOnError)
	dir := fs.String("dir", "charts", "Directory to clean")
	dryRun := fs.Bool("dry-run", false, "Show what would be deleted without deleting")
	fs.Parse(args)

	var errorFiles, validFiles, totalBytes int64

	err := walkDir(*dir, func(path string, info os.FileInfo) error {
		if info.IsDir() || !hasJSONExtension(path) {
			return nil
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		if isErrorResponse(data) {
			errorFiles++
			totalBytes += info.Size()
			if *dryRun {
				fmt.Printf("would delete: %s\n", path)
			} else {
				if err := os.Remove(path); err != nil {
					fmt.Fprintf(os.Stderr, "failed to delete %s: %v\n", path, err)
				}
			}
		} else {
			validFiles++
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Failed to walk directory: %v", err)
	}

	action := "deleted"
	if *dryRun {
		action = "would delete"
	}
	fmt.Printf("\n%s %d error files (%.2f MB), kept %d valid files\n",
		action, errorFiles, float64(totalBytes)/(1024*1024), validFiles)
}

func walkDir(dir string, fn func(path string, info os.FileInfo) error) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		path := dir + "/" + entry.Name()
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if err := fn(path, info); err != nil {
			return err
		}
		if entry.IsDir() {
			if err := walkDir(path, fn); err != nil {
				return err
			}
		}
	}
	return nil
}

func hasJSONExtension(path string) bool {
	return len(path) > 5 && path[len(path)-5:] == ".json"
}

func appendUnique(slice []int, val int) []int {
	for _, v := range slice {
		if v == val {
			return slice
		}
	}
	return append(slice, val)
}

// isErrorResponse checks if the API response is an error
func isErrorResponse(data []byte) bool {
	// Check for common error patterns
	// API returns [{"error":...}] for errors
	var response []struct {
		Error *struct {
			Json struct {
				Message string `json:"message"`
				Code    int    `json:"code"`
			} `json:"json"`
		} `json:"error,omitempty"`
		Result *struct{} `json:"result,omitempty"`
	}
	if err := json.Unmarshal(data, &response); err != nil {
		return false
	}
	if len(response) > 0 && response[0].Error != nil {
		return true
	}
	return false
}

// runConvert converts Stately XState format to sc proto-aligned JSON format
func runConvert(args []string) {
	fs := flag.NewFlagSet("convert", flag.ExitOnError)
	inputDir := fs.String("input", "charts", "Input directory with enriched Stately files")
	workers := fs.Int("workers", 4, "Number of concurrent workers")
	verbose := fs.Bool("v", false, "Verbose output")
	fs.Parse(args)

	// Collect all enriched files
	var files []string
	err := walkDir(*inputDir, func(path string, info os.FileInfo) error {
		if info.IsDir() || !hasJSONExtension(path) {
			return nil
		}
		// Quick check if file is enriched (has result wrapper)
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		if len(data) > 10 && string(data[:10]) == "[{\"result\"" {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		log.Fatalf("Failed to scan input directory: %v", err)
	}

	if len(files) == 0 {
		fmt.Println("No enriched files found. Run 'enrich' command first.")
		return
	}
	fmt.Printf("Found %d enriched files to convert\n", len(files))

	// Process files with worker pool
	var (
		converted atomic.Int64
		skipped   atomic.Int64
		errors    atomic.Int64
		wg        sync.WaitGroup
	)
	work := make(chan string, 100)

	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range work {
				// Output path: same location, but .scjson extension
				outPath := strings.TrimSuffix(path, ".json") + ".scjson"

				// Skip if already converted
				if _, err := os.Stat(outPath); err == nil {
					skipped.Add(1)
					continue
				}

				scs, err := convertStatelyFile(path)
				if err != nil {
					errors.Add(1)
					if *verbose {
						fmt.Fprintf(os.Stderr, "  error converting %s: %v\n", path, err)
					}
					continue
				}

				if len(scs) == 0 {
					skipped.Add(1)
					continue
				}

				// Write first statechart as single object (not array) for sc compatibility
				// If multiple machines, write additional files with _N suffix
				for idx, sc := range scs {
					var thisOutPath string
					if idx == 0 {
						thisOutPath = outPath
					} else {
						thisOutPath = strings.TrimSuffix(outPath, ".scjson") + fmt.Sprintf("_%d.scjson", idx)
					}

					data, err := json.MarshalIndent(sc, "", "  ")
					if err != nil {
						errors.Add(1)
						continue
					}
					if err := os.WriteFile(thisOutPath, data, 0644); err != nil {
						errors.Add(1)
						continue
					}

					converted.Add(1)
					if *verbose {
						fmt.Printf("  converted: %s\n", thisOutPath)
					}
				}
			}
		}()
	}

	// Feed work
	for _, f := range files {
		work <- f
	}
	close(work)
	wg.Wait()

	fmt.Printf("\nConverted %d files, skipped %d, %d errors\n", converted.Load(), skipped.Load(), errors.Load())
}

// StatelyData represents the enriched Stately editorData response
type StatelyData struct {
	Result struct {
		Data struct {
			JSON struct {
				Project struct {
					ID          string `json:"id"`
					Name        string `json:"name"`
					Description string `json:"description"`
				} `json:"project"`
				SelectedMachine *StatelyMachine `json:"selectedMachine"`
			} `json:"json"`
		} `json:"data"`
	} `json:"result"`
}

type StatelyMachine struct {
	ID         string            `json:"id"`
	Name       string            `json:"name"`
	Definition StatelyMachineDef `json:"definition"`
}

type StatelyMachineDef struct {
	ID       string          `json:"id"`
	Edges    []StatelyEdge   `json:"edges"`
	Context  json.RawMessage `json:"context"` // Can be map or string (JS code)
	RootNode StatelyNode     `json:"rootNode"`
	Schemas  *StatelySchemas `json:"schemas,omitempty"`
}

// GetContextMap returns context as a map, parsing JS code strings if needed
func (d *StatelyMachineDef) GetContextMap() map[string]any {
	if d.Context == nil || len(d.Context) == 0 {
		return nil
	}
	// Try as map first
	var m map[string]any
	if err := json.Unmarshal(d.Context, &m); err == nil {
		return m
	}
	// Try as string (JS code) - parse variable names from template
	var s string
	if err := json.Unmarshal(d.Context, &s); err == nil {
		return parseJSContextTemplate(s)
	}
	return nil
}

// parseJSContextTemplate extracts variable names from Stately's JS context templates
// Format: "{{({ input }) => ({ var1: value1, var2: value2, ... })}}"
func parseJSContextTemplate(s string) map[string]any {
	result := make(map[string]any)

	// Strip {{ and }} wrapper
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "{{") && strings.HasSuffix(s, "}}") {
		s = s[2 : len(s)-2]
	}

	// Find the return object: look for "=> ({" or "=> {" followed by "return {"
	// Simple approach: extract key names from "key:" or "key :" patterns
	// Skip function body complexity, just get variable names

	// Pattern 1: Arrow function returning object literal "=> ({ key1: ..., key2: ... })"
	if idx := strings.Index(s, "=> ("); idx != -1 {
		s = s[idx+4:] // skip "=> ("
	} else if idx := strings.Index(s, "=> {"); idx != -1 {
		// Could be "=> { return { ... } }" or just "=> { ... }"
		s = s[idx+3:]
	}

	// Extract variable names - look for "identifier:" patterns at start of line or after comma/brace
	// Use simple regex-like matching
	inKey := false
	keyStart := -1
	braceDepth := 0
	parenDepth := 0

	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '{':
			braceDepth++
			if braceDepth == 1 {
				inKey = true
				keyStart = -1
			}
		case '}':
			braceDepth--
		case '(':
			parenDepth++
		case ')':
			parenDepth--
		case ':':
			if braceDepth == 1 && parenDepth == 0 && keyStart >= 0 {
				key := strings.TrimSpace(s[keyStart:i])
				if isValidIdentifier(key) {
					result[key] = nil // We only extract names, not values
				}
				inKey = false
				keyStart = -1
			}
		case ',':
			if braceDepth == 1 && parenDepth == 0 {
				inKey = true
				keyStart = -1
			}
		case ' ', '\t', '\n', '\r':
			// whitespace
		default:
			if inKey && keyStart < 0 && isIdentifierStart(c) {
				keyStart = i
			}
		}
	}

	if len(result) == 0 {
		// Fallback: store raw code for reference
		result["__js_template__"] = s
	}

	return result
}

func isValidIdentifier(s string) bool {
	if len(s) == 0 {
		return false
	}
	for i, c := range s {
		if i == 0 {
			if !isIdentifierStart(byte(c)) {
				return false
			}
		} else {
			if !isIdentifierChar(byte(c)) {
				return false
			}
		}
	}
	return true
}

func isIdentifierStart(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '$'
}

func isIdentifierChar(c byte) bool {
	return isIdentifierStart(c) || (c >= '0' && c <= '9')
}

type StatelyEdge struct {
	ID     string `json:"id"`
	Source string `json:"source"`
	Target string `json:"target"`
	Data   struct {
		Actions       []StatelyAction   `json:"actions"`
		EventTypeData *StatelyEventType `json:"eventTypeData,omitempty"`
		Guard         *StatelyGuard     `json:"guard,omitempty"`
	} `json:"data"`
}

type StatelyEventType struct {
	Type      string `json:"type"` // "named", etc
	EventType string `json:"eventType"`
}

type StatelyGuard struct {
	Kind   string          `json:"kind"`   // "named"
	Type   string          `json:"type"`   // guard name
	Params json.RawMessage `json:"params"` // can be map, string, or array
}

type StatelyAction struct {
	Kind   string `json:"kind"` // "named"
	Action *struct {
		Type   string         `json:"type"`
		Params map[string]any `json:"params"`
	} `json:"action,omitempty"`
}

type StatelyNode struct {
	ID    string          `json:"id"`
	Data  StatelyNodeData `json:"data"`
	Nodes []StatelyNode   `json:"nodes"`
}

type StatelyNodeData struct {
	Key         string          `json:"key"`
	Initial     string          `json:"initial,omitempty"`
	Entry       []StatelyAction `json:"entry"`
	Exit        []StatelyAction `json:"exit"`
	Type        string          `json:"type,omitempty"` // "final", "parallel", "history", "annotation", "fail"
	History     string          `json:"history,omitempty"` // "shallow" or "deep"
	Description string          `json:"description,omitempty"`
	Tags        []string        `json:"tags,omitempty"`
	Invoke      json.RawMessage `json:"invoke,omitempty"`
	Invocations json.RawMessage `json:"invocations,omitempty"`
	MetaEntries json.RawMessage `json:"metaEntries,omitempty"`
}

type StatelySchemas struct {
	Events  map[string]any `json:"events"`
	Guards  map[string]any `json:"guards"`
	Actions map[string]any `json:"actions"`
	Context map[string]any `json:"context"`
}

// SCStatechart is our proto-aligned output format
type SCStatechart struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	RootState   SCState        `json:"root_state"`
	Transitions []SCTransition `json:"transitions"`
	Events      []SCEvent      `json:"events,omitempty"`
	Variables   map[string]any `json:"variables,omitempty"`
}

type SCState struct {
	Label        string         `json:"label"`
	Type         int            `json:"type"`                    // 1=BASIC, 2=OR, 3=AND
	Children     []SCState      `json:"children,omitempty"`
	IsInitial    bool           `json:"is_initial,omitempty"`
	IsFinal      bool           `json:"is_final,omitempty"`
	IsHistory    bool           `json:"is_history,omitempty"`    // True if history pseudostate
	HistoryType  int            `json:"history_type,omitempty"`  // 1=SHALLOW, 2=DEEP
	EntryActions []SCAction     `json:"entry_actions,omitempty"`
	ExitActions  []SCAction     `json:"exit_actions,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

type SCTransition struct {
	Label   string     `json:"label,omitempty"`
	From    []string   `json:"from"`
	To      []string   `json:"to"`
	Event   string     `json:"event,omitempty"`
	Guard   *SCGuard   `json:"guard,omitempty"`
	Actions []SCAction `json:"actions,omitempty"`
}

type SCGuard struct {
	Expression string `json:"expression,omitempty"`
	Language   string `json:"language,omitempty"`
}

type SCAction struct {
	Label      string         `json:"label,omitempty"`
	Expression string         `json:"expression,omitempty"`
	Language   string         `json:"language,omitempty"`
	Parameters map[string]any `json:"parameters,omitempty"`
}

type SCEvent struct {
	Label string `json:"label"`
}

// convertStatelyFile converts a single Stately enriched file to SC format
func convertStatelyFile(path string) ([]SCStatechart, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Parse the enriched format (array of results)
	var results []StatelyData
	if err := json.Unmarshal(data, &results); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var statecharts []SCStatechart
	for _, result := range results {
		if result.Result.Data.JSON.SelectedMachine == nil {
			continue
		}
		machine := *result.Result.Data.JSON.SelectedMachine
		sc := convertMachine(machine, result.Result.Data.JSON.Project.Name)
		statecharts = append(statecharts, sc)
	}

	return statecharts, nil
}

// convertMachine converts a single Stately machine to SC format
func convertMachine(m StatelyMachine, projectName string) SCStatechart {
	sc := SCStatechart{
		Name:      m.Name,
		Variables: m.Definition.GetContextMap(),
	}

	// Build state tree from rootNode
	sc.RootState = convertNode(m.Definition.RootNode, "")

	// Wrap in __root__ if needed
	if sc.RootState.Label != "__root__" {
		sc.RootState = SCState{
			Label:    "__root__",
			Type:     2, // OR
			Children: []SCState{sc.RootState},
		}
		sc.RootState.Children[0].IsInitial = true
	}

	// Convert edges to transitions
	eventsSeen := make(map[string]bool)
	for _, edge := range m.Definition.Edges {
		t := convertEdge(edge)
		sc.Transitions = append(sc.Transitions, t)
		if t.Event != "" {
			eventsSeen[t.Event] = true
		}
	}

	// Collect events
	for evt := range eventsSeen {
		sc.Events = append(sc.Events, SCEvent{Label: evt})
	}

	return sc
}

// convertNode recursively converts a Stately node to SC state
func convertNode(n StatelyNode, parentID string) SCState {
	s := SCState{
		Label: n.Data.Key,
		Metadata: map[string]any{
			"stately_id": n.ID,
		},
	}

	// Add description if present
	if n.Data.Description != "" {
		s.Metadata["description"] = n.Data.Description
	}

	// Determine state type and special flags
	switch n.Data.Type {
	case "parallel":
		s.Type = 3 // AND/PARALLEL
	case "history":
		s.IsHistory = true
		s.Type = 1 // BASIC — history pseudostates have no children
		if n.Data.History == "deep" {
			s.HistoryType = 2 // HISTORY_TYPE_DEEP
		} else {
			s.HistoryType = 1 // HISTORY_TYPE_SHALLOW (default)
		}
	case "final", "fail":
		s.IsFinal = true
		if n.Data.Type == "fail" {
			s.Metadata["stately_type"] = "fail"
		}
		if len(n.Nodes) > 0 {
			s.Type = 2
		} else {
			s.Type = 1
		}
	case "annotation":
		// Annotations are documentation nodes, not real states — skip as metadata
		s.Type = 1
		s.Metadata["stately_type"] = "annotation"
	default:
		if len(n.Nodes) > 0 {
			s.Type = 2 // OR/NORMAL
		} else {
			s.Type = 1 // BASIC
		}
	}

	// Preserve tags
	if len(n.Data.Tags) > 0 {
		s.Metadata["tags"] = n.Data.Tags
	}

	// Preserve invocations/invoke as metadata
	if len(n.Data.Invoke) > 0 && string(n.Data.Invoke) != "null" && string(n.Data.Invoke) != "[]" {
		var invocations any
		if err := json.Unmarshal(n.Data.Invoke, &invocations); err == nil {
			s.Metadata["invoke"] = invocations
		}
	}
	if len(n.Data.Invocations) > 0 && string(n.Data.Invocations) != "null" && string(n.Data.Invocations) != "[]" {
		var invocations any
		if err := json.Unmarshal(n.Data.Invocations, &invocations); err == nil {
			s.Metadata["invocations"] = invocations
		}
	}

	// Preserve meta entries
	if len(n.Data.MetaEntries) > 0 && string(n.Data.MetaEntries) != "null" {
		var meta any
		if err := json.Unmarshal(n.Data.MetaEntries, &meta); err == nil {
			s.Metadata["meta_entries"] = meta
		}
	}

	// Convert children
	for _, child := range n.Nodes {
		childState := convertNode(child, n.ID)
		// Mark initial state
		if n.Data.Initial != "" && child.Data.Key == n.Data.Initial {
			childState.IsInitial = true
		}
		s.Children = append(s.Children, childState)
	}

	// Convert entry/exit actions
	for _, action := range n.Data.Entry {
		s.EntryActions = append(s.EntryActions, convertAction(action))
	}
	for _, action := range n.Data.Exit {
		s.ExitActions = append(s.ExitActions, convertAction(action))
	}

	return s
}

// convertEdge converts a Stately edge to SC transition
func convertEdge(e StatelyEdge) SCTransition {
	t := SCTransition{
		From: []string{extractStateKey(e.Source)},
		To:   []string{extractStateKey(e.Target)},
	}

	// Extract event
	if e.Data.EventTypeData != nil && e.Data.EventTypeData.EventType != "" {
		t.Event = e.Data.EventTypeData.EventType
	}

	// Extract guard
	if e.Data.Guard != nil && e.Data.Guard.Type != "" {
		t.Guard = &SCGuard{
			Expression: e.Data.Guard.Type,
			Language:   "xstate",
		}
	}

	// Convert actions
	for _, action := range e.Data.Actions {
		t.Actions = append(t.Actions, convertAction(action))
	}

	return t
}

// convertAction converts a Stately action to SC action
func convertAction(a StatelyAction) SCAction {
	action := SCAction{
		Language: "xstate",
	}
	if a.Action != nil {
		action.Label = a.Action.Type
		action.Parameters = a.Action.Params
	}
	return action
}

// extractStateKey extracts the last part of a dotted state ID
// e.g., "invoiceTimeline.idle" -> "idle"
func extractStateKey(stateID string) string {
	parts := strings.Split(stateID, ".")
	if len(parts) > 0 {
		return parts[len(parts)-1]
	}
	return stateID
}
