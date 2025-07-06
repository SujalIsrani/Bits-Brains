import { KeyEvent, TypingSession } from '../models/typing-session';

export class TypingAnalyzer {
  private keyEvents: KeyEvent[] = [];
  private sessionStartTime: number = 0;
  private lastKeyReleaseTime: number = 0;
  private backspaceCount: number = 0;
  private totalKeystrokes: number = 0;

  startSession(): void {
    this.keyEvents = [];
    this.sessionStartTime = Date.now();
    this.lastKeyReleaseTime = 0;
    this.backspaceCount = 0;
    this.totalKeystrokes = 0;
  }

  recordKeyEvent(key: string, pressTime: number, releaseTime: number): void {
    const holdTime = releaseTime - pressTime;
    const flightTime = this.lastKeyReleaseTime > 0 ? pressTime - this.lastKeyReleaseTime : 0;

    const keyEvent: KeyEvent = {
      key,
      pressTime,
      releaseTime,
      holdTime,
      flightTime
    };

    this.keyEvents.push(keyEvent);
    this.lastKeyReleaseTime = releaseTime;
    this.totalKeystrokes++;

    if (key === 'Backspace') {
      this.backspaceCount++;
    }
  }

  calculateMetrics(): any {
    if (this.keyEvents.length === 0) {
      return null;
    }

    const holdTimes = this.keyEvents.map(e => e.holdTime);
    const flightTimes = this.keyEvents.filter(e => e.flightTime > 0).map(e => e.flightTime);

    const meanHoldTime = this.calculateMean(holdTimes);
    const stdHoldTime = this.calculateStandardDeviation(holdTimes);
    const holdTimeVariability = meanHoldTime > 0 ? stdHoldTime / meanHoldTime : 0;

    const meanFlightTime = this.calculateMean(flightTimes);
    const stdFlightTime = this.calculateStandardDeviation(flightTimes);

    const backspaceRate = this.totalKeystrokes > 0 ? (this.backspaceCount / this.totalKeystrokes) * 100 : 0;
    
    // Calculate pause frequency (pauses >500ms per 100 keys)
    const longPauses = flightTimes.filter(ft => ft > 500).length;
    const pauseFrequency = this.totalKeystrokes > 0 ? (longPauses / this.totalKeystrokes) * 100 : 0;

    const sessionDuration = (Date.now() - this.sessionStartTime) / 1000; // in seconds
    const typingSpeed = sessionDuration > 0 ? (this.totalKeystrokes / sessionDuration) * 60 : 0; // keys per minute

    return {
      sessionDuration,
      totalKeystrokes: this.totalKeystrokes,
      typingSpeed,
      meanHoldTime,
      stdHoldTime,
      holdTimeVariability,
      meanFlightTime,
      stdFlightTime,
      backspaceRate,
      pauseFrequency,
      keyEvents: this.keyEvents
    };
  }

  private calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private calculateStandardDeviation(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = this.calculateMean(values);
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return Math.sqrt(this.calculateMean(squaredDiffs));
  }

  getSessionDuration(): number {
    return (Date.now() - this.sessionStartTime) / 1000;
  }

  getTotalKeystrokes(): number {
    return this.totalKeystrokes;
  }

  getCurrentTypingSpeed(): number {
    const duration = this.getSessionDuration();
    return duration > 0 ? (this.totalKeystrokes / duration) * 60 : 0;
  }
}