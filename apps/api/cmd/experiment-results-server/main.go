// Isolated loopback experiment archive/readback; does not open the store database.
package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"storefront/api/internal/experimentresults"
	"syscall"
	"time"
)

func main() {
	var repo experimentresults.Repository
	if dsn := os.Getenv("TIGER_DATABASE_URL"); dsn != "" {
		postgres, err := experimentresults.NewPostgres(dsn)
		if err != nil {
			fmt.Fprintln(os.Stderr, "invalid Tiger configuration (details redacted)")
			os.Exit(1)
		}
		repo = postgres
	}
	root := os.Getenv("EXPERIMENT_RESULTS_DIR")
	if root == "" {
		root = "../../data/experiment-results"
	}
	service, err := experimentresults.New(root, repo)
	if err != nil {
		fmt.Fprintln(os.Stderr, "local experiment archive unavailable")
		os.Exit(1)
	}
	defer service.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:4311")
	if err != nil {
		fmt.Fprintln(os.Stderr, "experiment receiver port unavailable")
		os.Exit(1)
	}
	mux := http.NewServeMux()
	service.Register(mux)
	server := http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second, WriteTimeout: 15 * time.Second, IdleTimeout: 30 * time.Second}
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}()
	fmt.Println("Experiment results receiver ready on loopback port 4311")
	if err = server.Serve(listener); err != nil && err != http.ErrServerClosed {
		fmt.Fprintln(os.Stderr, "experiment receiver stopped unexpectedly")
		os.Exit(1)
	}
}
