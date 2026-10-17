package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

// O_EXCL protects one data directory; stale PIDs may be recovered after a crash.
func acquireLock(dataDir string) (func(), error) {
	if err := os.MkdirAll(dataDir, 0700); err != nil {
		return nil, err
	}
	file := filepath.Join(dataDir, ".active.lock")
	for attempt := 0; attempt < 2; attempt++ {
		handle, err := os.OpenFile(file, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err == nil {
			if _, err = fmt.Fprint(handle, os.Getpid()); err != nil {
				handle.Close()
				os.Remove(file)
				return nil, err
			}
			handle.Close()
			return func() { _ = os.Remove(file) }, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, err
		}
		body, err := os.ReadFile(file)
		if err != nil {
			return nil, err
		}
		pid, err := strconv.Atoi(strings.TrimSpace(string(body)))
		if err != nil || pid <= 0 {
			return nil, fmt.Errorf("invalid data lock at %s; inspect and remove it after stopping the API", file)
		}
		if err = syscall.Kill(pid, 0); !errors.Is(err, syscall.ESRCH) {
			return nil, fmt.Errorf("local data is in use by PID %d; stop the API before seed/reset or another API instance", pid)
		}
		if err = os.Remove(file); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("could not acquire local data lock")
}
