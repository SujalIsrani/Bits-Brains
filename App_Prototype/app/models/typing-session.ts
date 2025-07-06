export interface TypingSession {
  sessionId: string;
  patientId: string;
  diagnosis: string;
  updrsScore: number;
  tremorSeverity: number;
  medicationStatus: string;
  timestamp: string;
  sessionDuration: number;
  totalKeystrokes: number;
  typingSpeed: number;
  meanHoldTime: number;
  stdHoldTime: number;
  holdTimeVariability: number;
  meanFlightTime: number;
  stdFlightTime: number;
  backspaceRate: number;
  pauseFrequency: number;
  keyEvents: KeyEvent[];
}

export interface KeyEvent {
  key: string;
  pressTime: number;
  releaseTime: number;
  holdTime: number;
  flightTime: number;
}

export interface UserMetadata {
  patientId: string;
  diagnosis: 'Parkinson\'s' | 'Healthy';
  updrsScore: number;
  tremorSeverity: number;
  medicationStatus: 'On' | 'Off' | 'NA';
}