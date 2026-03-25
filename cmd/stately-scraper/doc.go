// Command stately-scraper fetches and manages Stately project data for local
// analysis and conversion.
//
// The command is organized as subcommands. Some operations call authenticated
// Stately endpoints and accept either -cookie or the STATELY_COOKIE
// environment variable. The tool can list accessible projects, discover chart
// metadata, download machine definitions, enrich local files with additional
// editor data, and convert enriched files into formats used by this repository.
//
// Usage:
//
//	stately-scraper <command> [flags]
//
// Commands:
//
//	projects   List visible personal and team projects.
//	discover   Search or paginate projects and optionally download charts.
//	asset      Fetch a single asset by URL.
//	stats      Summarize or verify a local chart directory.
//	cleanup    Remove generated files that are no longer useful.
//	enrich     Fetch fuller machine data for previously discovered files.
//	convert    Convert enriched Stately files into local forms.
//
// Examples:
//
//	stately-scraper projects -cookie "$STATELY_COOKIE"
//	stately-scraper discover -download -output charts -all
//	stately-scraper enrich -input charts -workers 8
//	stately-scraper stats -dir charts -verify
package main
