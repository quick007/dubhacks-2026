// Isolated loopback receiver for local verification; production hosting is separate.
package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"storefront/api/internal/shoppingevents"
	"syscall"
	"time"
)

func main() {
	config, err := shoppingevents.ConfigFromEnv()
	if err != nil || !config.Enabled {
		fmt.Fprintln(os.Stderr, "explicit shopping telemetry configuration required")
		os.Exit(1)
	}
	var repo shoppingevents.Repository
	if dsn := os.Getenv("TIGER_DATABASE_URL"); dsn != "" {
		repo, err = shoppingevents.NewPostgres(dsn)
		if err != nil {
			fmt.Fprintln(os.Stderr, "invalid Tiger configuration (details redacted)")
			os.Exit(1)
		}
	}
	root := os.Getenv("STOREFRONT_DATA_DIR")
	if root == "" {
		root = "../../data"
	}
	service, err := shoppingevents.New(filepath.Join(root, "shopping-events"), config, repo)
	if err != nil {
		fmt.Fprintln(os.Stderr, "shopping archive unavailable")
		os.Exit(1)
	}
	defer service.Close()
	port := os.Getenv("PORT")
	if port == "" {
		port = "4330"
	}
	listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", port))
	if err != nil {
		fmt.Fprintln(os.Stderr, "shopping receiver loopback port unavailable")
		os.Exit(1)
	}
	mux := http.NewServeMux()
	service.Register(mux)
	server := http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 10 * time.Second, IdleTimeout: 30 * time.Second}
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-signals
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}()
	fmt.Println("Shopping receiver ready on loopback port " + port)
	if err = server.Serve(listener); err != nil && err != http.ErrServerClosed {
		fmt.Fprintln(os.Stderr, "shopping receiver stopped unexpectedly")
	}
}
