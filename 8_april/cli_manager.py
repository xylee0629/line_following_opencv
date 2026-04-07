# cli_manager.py
import sys
import threading

class TerminalController:
    def __init__(self):
        self.app_running = True
        self.robot_active = False
        self.thread = threading.Thread(target=self._listener, daemon=True)

    def start(self):
        self.thread.start()

    def _listener(self):
        print("\n--- Terminal Control Active ---")
        print("Type 's' + Enter to toggle Start/Pause")
        print("Type 'q' + Enter to Quit\n")
        
        while self.app_running:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 's':
                self.robot_active = not self.robot_active
                state = "STARTED" if self.robot_active else "PAUSED"
                print(f"\n[COMMAND] Motors {state}!")
            elif cmd == 'q':
                print("\n[COMMAND] Shutting down...")
                self.app_running = False