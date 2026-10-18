// Isolated local fake-order receiver; no store database or hosted connection.
package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"storefront/api/internal/democheckout"
	"syscall"
	"time"
)

func main() {
	root := os.Getenv("DEMO_CHECKOUT_DIR")
	if root == "" {
		root = "../../data/demo-checkout-cohort-v1"
	}
	service, err := democheckout.New(filepath.Clean(root))
	if err != nil {
		fmt.Fprintln(os.Stderr, "local demo receipt archive unavailable")
		os.Exit(1)
	}
	port := os.Getenv("PORT")
	if port == "" {
		port = "4331"
	}
	listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", port))
	if err != nil {
		fmt.Fprintln(os.Stderr, "local demo receipt port unavailable")
		os.Exit(1)
	}
	mux := http.NewServeMux()
	service.Register(mux)
	server := http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 10 * time.Second, IdleTimeout: 30 * time.Second}
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}()
	fmt.Println("Local-only demo checkout ready on loopback port " + port)
	if err = server.Serve(listener); err != nil && err != http.ErrServerClosed {
		fmt.Fprintln(os.Stderr, "demo checkout receiver stopped unexpectedly")
	}
}
