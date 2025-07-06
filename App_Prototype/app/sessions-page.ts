import { EventData, Page, ListView, Button, confirm } from '@nativescript/core';
import { DataStorage } from './services/data-storage';
import { TypingSession } from './models/typing-session';

let page: Page;
let listView: ListView;
let sessions: TypingSession[] = [];

export function navigatingTo(args: EventData) {
  page = <Page>args.object;
  listView = page.getViewById('sessionsList') as ListView;
  
  loadSessions();
}

function loadSessions() {
  sessions = DataStorage.getAllSessions();
  listView.items = sessions.map(session => ({
    sessionId: session.sessionId,
    timestamp: new Date(session.timestamp.replace(/-/g, ':')).toLocaleString(),
    duration: `${session.sessionDuration.toFixed(1)}s`,
    keystrokes: session.totalKeystrokes,
    speed: `${session.typingSpeed.toFixed(1)} keys/min`,
    holdTime: `${session.meanHoldTime.toFixed(1)}ms`,
    flightTime: `${session.meanFlightTime.toFixed(1)}ms`,
    backspaceRate: `${session.backspaceRate.toFixed(1)}%`
  }));
}

export function onSessionTap(args: any) {
  const session = sessions[args.index];
  const message = `
Session Details:
Patient ID: ${session.patientId}
Diagnosis: ${session.diagnosis}
UPDRS Score: ${session.updrsScore}
Tremor Severity: ${session.tremorSeverity}
Medication Status: ${session.medicationStatus}

Session Metrics:
Duration: ${session.sessionDuration.toFixed(1)}s
Total Keystrokes: ${session.totalKeystrokes}
Typing Speed: ${session.typingSpeed.toFixed(1)} keys/min
Mean Hold Time: ${session.meanHoldTime.toFixed(1)}ms ± ${session.stdHoldTime.toFixed(1)}ms
Hold Time Variability: ${session.holdTimeVariability.toFixed(4)}
Mean Flight Time: ${session.meanFlightTime.toFixed(1)}ms ± ${session.stdFlightTime.toFixed(1)}ms
Backspace Rate: ${session.backspaceRate.toFixed(1)}%
Pause Frequency: ${session.pauseFrequency.toFixed(1)}%
  `;
  
  alert(message);
}

export function onClearData() {
  confirm({
    title: 'Clear All Data',
    message: 'Are you sure you want to delete all session data? This action cannot be undone.',
    okButtonText: 'Delete All',
    cancelButtonText: 'Cancel'
  }).then(result => {
    if (result) {
      DataStorage.clearAllData();
      loadSessions();
      alert('All data has been cleared.');
    }
  });
}

export function onBack() {
  page.frame.goBack();
}