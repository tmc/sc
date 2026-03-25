package main

import (
	"net/http"
	"time"
)

// NewClient creates a new HTTP client with default headers.
func NewClient(cookie string, sleepDuration time.Duration) *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.MaxIdleConnsPerHost = 100 // Allow more idle connections for workers
	return &http.Client{
		Transport: &headerTransport{
			Transport:     t,
			Cookie:        cookie,
			SleepDuration: sleepDuration,
		},
	}
}

type headerTransport struct {
	Transport     http.RoundTripper
	Cookie        string
	SleepDuration time.Duration
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.SleepDuration > 0 {
		time.Sleep(t.SleepDuration)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36")
	req.Header.Set("sec-ch-ua", "\"Brave\";v=\"143\", \"Chromium\";v=\"143\", \"Not A(Brand\";v=\"24\"")
	req.Header.Set("sec-ch-ua-mobile", "?0")
	req.Header.Set("sec-ch-ua-platform", "\"macOS\"")

	if t.Cookie != "" {
		req.Header.Set("Cookie", t.Cookie)
	}

	return t.Transport.RoundTrip(req)
}
