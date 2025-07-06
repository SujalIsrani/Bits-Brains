import { EventData, Page, Button, Label, TextView, GridLayout } from '@nativescript/core';
import { TypingAnalyzer } from './services/typing-analyzer';
import { DataStorage } from './services/data-storage';
import { TypingSession } from './models/typing-session';

let page: Page;
let typingAnalyzer: TypingAnalyzer;
let sessionTimer: any;
let sessionStartTime: number;
let isSessionActive: boolean = false;
let timerLabel: Label;
let keystrokeLabel: Label;
let speedLabel: Label;
let typingArea: TextView;
let startButton: Button;
let doneButton: Button;
let keyboardContainer: GridLayout;

const MINIMUM_SESSION_DURATION = 60; // 60 seconds

export function navigatingTo(args: EventData) {
  page = <Page>args.object;
  
  timerLabel = page.getViewById('timerLabel') as Label;
  keystrokeLabel = page.getViewById('keystrokeLabel') as Label;
  speedLabel = page.getViewById('speedLabel') as Label;
  typingArea = page.getViewById('typingArea') as TextView;
  startButton = page.getViewById('startButton') as Button;
  doneButton = page.getViewById('doneButton') as Button;
  keyboardContainer = page.getViewById('keyboardContainer') as GridLayout;

  typingAnalyzer = new TypingAnalyzer();
  updateUI();
}

export function onStartSession() {
  if (isSessionActive) return;

  isSessionActive = true;
  sessionStartTime = Date.now();
  typingAnalyzer.startSession();
  
  startButton.isEnabled = false;
  doneButton.isEnabled = false;
  doneButton.text = 'Complete Session (60s minimum)';
  
  typingArea.text = '';
  typingArea.isEnabled = true;
  
  // Start timer
  sessionTimer = setInterval(() => {
    updateTimer();
  }, 1000);
  
  updateUI();
}

export function onEndSession() {
  if (!isSessionActive) return;

  const sessionDuration = (Date.now() - sessionStartTime) / 1000;
  
  if (sessionDuration < MINIMUM_SESSION_DURATION) {
    alert(`Session must be at least ${MINIMUM_SESSION_DURATION} seconds. Current: ${Math.floor(sessionDuration)}s`);
    return;
  }

  isSessionActive = false;
  clearInterval(sessionTimer);
  
  const metrics = typingAnalyzer.calculateMetrics();
  if (metrics) {
    saveSession(metrics);
  }
  
  startButton.isEnabled = true;
  doneButton.isEnabled = true;
  doneButton.text = 'End Session';
  typingArea.isEnabled = false;
  
  updateUI();
  
  // Show results
  showSessionResults(metrics);
}

function updateTimer() {
  const elapsed = (Date.now() - sessionStartTime) / 1000;
  const remaining = Math.max(0, MINIMUM_SESSION_DURATION - elapsed);
  
  if (remaining <= 0) {
    doneButton.isEnabled = true;
    doneButton.text = 'Complete Session';
    timerLabel.text = `Session Time: ${Math.floor(elapsed)}s`;
  } else {
    timerLabel.text = `Time Remaining: ${Math.floor(remaining)}s`;
  }
  
  updateUI();
}

function updateUI() {
  if (isSessionActive) {
    keystrokeLabel.text = `Keystrokes: ${typingAnalyzer.getTotalKeystrokes()}`;
    speedLabel.text = `Speed: ${typingAnalyzer.getCurrentTypingSpeed().toFixed(1)} keys/min`;
  } else {
    keystrokeLabel.text = 'Keystrokes: 0';
    speedLabel.text = 'Speed: 0.0 keys/min';
    timerLabel.text = 'Ready to start';
  }
}

function saveSession(metrics: any) {
  const userMetadata = DataStorage.getUserMetadata();
  if (!userMetadata) {
    alert('User metadata not found. Please restart the app.');
    return;
  }

  const session: TypingSession = {
    sessionId: `${userMetadata.patientId}_${Date.now()}`,
    patientId: userMetadata.patientId,
    diagnosis: userMetadata.diagnosis,
    updrsScore: userMetadata.updrsScore,
    tremorSeverity: userMetadata.tremorSeverity,
    medicationStatus: userMetadata.medicationStatus,
    timestamp: new Date().toISOString().replace(/[:.]/g, '-'),
    sessionDuration: metrics.sessionDuration,
    totalKeystrokes: metrics.totalKeystrokes,
    typingSpeed: metrics.typingSpeed,
    meanHoldTime: metrics.meanHoldTime,
    stdHoldTime: metrics.stdHoldTime,
    holdTimeVariability: metrics.holdTimeVariability,
    meanFlightTime: metrics.meanFlightTime,
    stdFlightTime: metrics.stdFlightTime,
    backspaceRate: metrics.backspaceRate,
    pauseFrequency: metrics.pauseFrequency,
    keyEvents: metrics.keyEvents
  };

  DataStorage.saveSession(session);
}

function showSessionResults(metrics: any) {
  const message = `
Session Complete!

Duration: ${metrics.sessionDuration.toFixed(1)}s
Total Keystrokes: ${metrics.totalKeystrokes}
Typing Speed: ${metrics.typingSpeed.toFixed(1)} keys/min
Mean Hold Time: ${metrics.meanHoldTime.toFixed(1)}ms
Mean Flight Time: ${metrics.meanFlightTime.toFixed(1)}ms
Backspace Rate: ${metrics.backspaceRate.toFixed(1)}%
Pause Frequency: ${metrics.pauseFrequency.toFixed(1)}%

Session saved successfully!
  `;
  
  alert(message);
}

export function onKeyPress(args: any) {
  if (!isSessionActive) return;
  
  const button = args.object as Button;
  const key = button.text;
  const pressTime = Date.now();
  
  // Simulate key press/release for demonstration
  setTimeout(() => {
    const releaseTime = Date.now();
    typingAnalyzer.recordKeyEvent(key, pressTime, releaseTime);
    
    // Update typing area
    if (key === 'Backspace') {
      const currentText = typingArea.text || '';
      typingArea.text = currentText.slice(0, -1);
    } else if (key === 'Space') {
      typingArea.text = (typingArea.text || '') + ' ';
    } else if (key.length === 1) {
      typingArea.text = (typingArea.text || '') + key;
    }
    
    updateUI();
  }, 50 + Math.random() * 100); // Simulate variable hold time
}

export function onViewSessions() {
  page.frame.navigate('sessions-page');
}

export function onExportData() {
  page.frame.navigate('export-page');
}