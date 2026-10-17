package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func dataDirectory() (string, error) {
	directory := os.Getenv("STOREFRONT_DATA_DIR")
	if directory == "" {
		directory = "../../data"
	}
	return filepath.Abs(directory)
}
func main() {
	if len(os.Args) > 1 {
		if err := maintenance(os.Args[1]); err != nil {
			log.Fatal(err)
		}
		return
	}
	listener, err := net.Listen("tcp", "127.0.0.1:4311")
	if err != nil {
		log.Fatalf("Cannot start API on 127.0.0.1:4311; stop the process using that port: %v", err)
	}
	app, err := NewApp(AppOptions{})
	if err != nil {
		listener.Close()
		log.Fatal(err)
	}
	defer app.Close()
	server := &http.Server{Handler: app, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 15 * time.Second, IdleTimeout: 60 * time.Second}
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
	}()
	log.Println("Synthetic storefront API listening on http://127.0.0.1:4311")
	if err = server.Serve(listener); err != nil && err != http.ErrServerClosed {
		log.Print(err)
	}
}
func maintenance(command string) error {
	if command != "seed" && command != "reset" {
		return fmt.Errorf("usage: go run . [seed|reset]")
	}
	directory, err := dataDirectory()
	if err != nil {
		return err
	}
	cwd, _ := os.Getwd()
	home, _ := os.UserHomeDir()
	workspace, _ := filepath.Abs("../..")
	if directory == string(os.PathSeparator) || directory == cwd || directory == home || directory == workspace {
		return fmt.Errorf("refusing unsafe local data directory %s", directory)
	}
	release, err := acquireLock(directory)
	if err != nil {
		return err
	}
	defer release()
	if command == "reset" {
		for _, file := range []string{"metadata.sqlite", "metadata.sqlite-wal", "metadata.sqlite-shm", "artifacts"} {
			if err = os.RemoveAll(filepath.Join(directory, file)); err != nil {
				return err
			}
		}
	}
	store, err := openStore(directory)
	if err != nil {
		return err
	}
	defer store.db.Close()
	fmt.Printf("Local demo %s complete at %s\n", command, directory)
	return nil
}
