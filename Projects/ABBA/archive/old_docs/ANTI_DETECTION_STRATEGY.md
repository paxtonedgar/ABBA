# Anti-Detection Strategy for Bet Placement with BrowserBase

## Overview

This document outlines the comprehensive anti-detection strategy for placing bets on FanDuel and DraftKings using BrowserBase, designed to avoid detection while maintaining operational effectiveness.

## Core Anti-Detection Principles

### 1. Browser Fingerprint Randomization

**BrowserBase Stealth Configuration:**
```python
stealth_config = {
    "enabled": True,
    "viewport": {
        "width": random.randint(1820, 2020),  # Randomized viewport
        "height": random.randint(1030, 1130)
    },
    "userAgent": random.choice(user_agents),  # Rotating user agents
    "webgl": {
        "enabled": True,
        "vendor": "Google Inc. (Intel)",
        "renderer": "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0, D3D11)"
    },
    "canvas": {
        "enabled": True,
        "noise": True  # Canvas fingerprint randomization
    },
    "audio": {
        "enabled": True,
        "noise": True  # Audio fingerprint randomization
    },
    "timezone": {
        "enabled": True,
        "value": random.choice(timezones)
    },
    "geolocation": {
        "enabled": True,
        "latitude": random.uniform(40.0, 45.0),
        "longitude": random.uniform(-75.0, -70.0)
    }
}
```

### 2. Human-Like Behavior Patterns

**Typing Patterns:**
- Random delays between keystrokes (50-150ms)
- Variable typing speed
- Occasional typos and corrections
- Natural pause patterns

**Mouse Movement:**
- Curved mouse paths (not straight lines)
- Variable movement speeds
- Natural acceleration/deceleration
- Occasional hover patterns

**Click Patterns:**
- Random delays before/after clicks (0.5-3.0s)
- Variable click durations
- Natural double-click patterns
- Context-appropriate click locations

### 3. Session Management

**Session Rotation:**
- Rotate sessions every 1 hour
- Random session duration (45-75 minutes)
- Clean session state between rotations
- Different fingerprint per session

**Cookie Management:**
- Preserve essential cookies
- Randomize non-essential cookies
- Natural cookie expiration patterns
- Platform-specific cookie handling

### 4. Network Behavior

**Request Patterns:**
- Natural request timing
- Variable request intervals
- Realistic referrer headers
- Platform-appropriate user agents

**Proxy Strategy:**
- Residential proxy rotation
- Geographic consistency
- Bandwidth throttling simulation
- Connection stability patterns

## Platform-Specific Strategies

### FanDuel Anti-Detection

**Login Patterns:**
```python
fanduel_login_flow = [
    {"action": "navigate", "delay": "2-4s"},
    {"action": "wait_for_page_load", "delay": "1-2s"},
    {"action": "type_username", "delay": "0.5-1.5s", "typing_speed": "50-150ms"},
    {"action": "pause", "delay": "1-3s"},
    {"action": "type_password", "delay": "0.5-1.5s", "typing_speed": "50-150ms"},
    {"action": "pause", "delay": "1-2s"},
    {"action": "click_login", "delay": "0.5-1s"},
    {"action": "wait_for_redirect", "delay": "3-6s"}
]
```

**Bet Placement Patterns:**
```python
fanduel_bet_flow = [
    {"action": "navigate_to_event", "delay": "2-4s"},
    {"action": "scroll_to_market", "delay": "1-2s"},
    {"action": "click_market", "delay": "1-2s"},
    {"action": "wait_for_odds", "delay": "1-3s"},
    {"action": "type_stake", "delay": "1-2s", "typing_speed": "100-200ms"},
    {"action": "verify_odds", "delay": "1-2s"},
    {"action": "click_place_bet", "delay": "1-2s"},
    {"action": "wait_for_confirmation", "delay": "3-8s"}
]
```

### DraftKings Anti-Detection

**Login Patterns:**
```python
draftkings_login_flow = [
    {"action": "navigate", "delay": "2-4s"},
    {"action": "wait_for_page_load", "delay": "1-2s"},
    {"action": "type_username", "delay": "0.5-1.5s", "typing_speed": "50-150ms"},
    {"action": "pause", "delay": "1-3s"},
    {"action": "type_password", "delay": "0.5-1.5s", "typing_speed": "50-150ms"},
    {"action": "pause", "delay": "1-2s"},
    {"action": "click_login", "delay": "0.5-1s"},
    {"action": "wait_for_redirect", "delay": "3-6s"}
]
```

**Bet Placement Patterns:**
```python
draftkings_bet_flow = [
    {"action": "navigate_to_event", "delay": "2-4s"},
    {"action": "scroll_to_market", "delay": "1-2s"},
    {"action": "click_market", "delay": "1-2s"},
    {"action": "wait_for_odds", "delay": "1-3s"},
    {"action": "type_stake", "delay": "1-2s", "typing_speed": "100-200ms"},
    {"action": "verify_odds", "delay": "1-2s"},
    {"action": "click_place_bet", "delay": "1-2s"},
    {"action": "wait_for_confirmation", "delay": "3-8s"}
]
```

## Advanced Anti-Detection Techniques

### 1. Behavioral Biometrics Evasion

**Mouse Movement Patterns:**
- Implement Bezier curve movements
- Variable acceleration profiles
- Natural pause patterns
- Context-aware movement speeds

**Keyboard Dynamics:**
- Variable typing speeds
- Natural error patterns
- Context-appropriate delays
- Platform-specific typing habits

### 2. Machine Learning Evasion

**Feature Randomization:**
- Randomize timing patterns
- Vary interaction sequences
- Natural error introduction
- Context-aware behavior

**Pattern Avoidance:**
- Avoid repetitive sequences
- Introduce natural variations
- Context-appropriate actions
- Human-like decision making

### 3. Network Fingerprint Evasion

**HTTP Headers:**
```python
headers = {
    "User-Agent": random.choice(user_agents),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": random.choice(["1", None]),
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0"
}
```

**TLS Fingerprint:**
- Use realistic TLS configurations
- Match platform expectations
- Avoid automation indicators
- Natural certificate handling

## Risk Mitigation Strategies

### 1. Account Protection

**Betting Patterns:**
- Vary bet sizes naturally
- Mix bet types and markets
- Avoid suspicious patterns
- Natural timing intervals

**Session Management:**
- Regular session rotation
- Clean session state
- Natural logout patterns
- Platform-appropriate behavior

### 2. Error Handling

**Graceful Degradation:**
- Handle detection gracefully
- Implement fallback strategies
- Natural error responses
- Platform-specific handling

**Recovery Mechanisms:**
- Automatic session rotation
- Error pattern analysis
- Adaptive behavior adjustment
- Learning from failures

### 3. Monitoring and Alerts

**Detection Monitoring:**
- Track detection indicators
- Monitor account status
- Analyze error patterns
- Adaptive response strategies

**Performance Metrics:**
- Success rate tracking
- Detection rate monitoring
- Response time analysis
- Pattern effectiveness

## Implementation Guidelines

### 1. Configuration Management

**Environment Variables:**
```bash
BROWSERBASE_API_KEY=your_api_key
BROWSERBASE_PROJECT_ID=your_project_id
BROWSERBASE_PROXY=your_proxy_url
FANDUEL_USERNAME=your_username
FANDUEL_PASSWORD=your_password
DRAFTKINGS_USERNAME=your_username
DRAFTKINGS_PASSWORD=your_password
```

**Configuration File:**
```yaml
security:
  browserbase:
    enabled: true
    stealth_mode: true
    session_rotation_frequency: 3600
    proxy_enabled: false
  anti_detection:
    human_like_behavior: true
    fingerprint_randomization: true
    session_rotation: true
    behavior_patterns: true
```

### 2. Testing and Validation

**Detection Testing:**
- Regular detection tests
- Platform-specific validation
- Behavior pattern analysis
- Performance monitoring

**Success Metrics:**
- Bet placement success rate
- Detection avoidance rate
- Session longevity
- Platform compatibility

### 3. Maintenance and Updates

**Regular Updates:**
- Platform selector updates
- Behavior pattern refinement
- Detection method adaptation
- Performance optimization

**Monitoring:**
- Real-time performance tracking
- Detection pattern analysis
- Platform change monitoring
- Adaptive strategy updates

## Best Practices

### 1. Operational Security

**Credential Management:**
- Secure credential storage
- Regular credential rotation
- Platform-specific security
- Access control management

**Session Security:**
- Secure session handling
- Clean session termination
- Data sanitization
- Privacy protection

### 2. Performance Optimization

**Resource Management:**
- Efficient session usage
- Optimal timing patterns
- Resource cleanup
- Performance monitoring

**Scalability:**
- Concurrent session support
- Load balancing
- Resource optimization
- Performance scaling

### 3. Compliance and Ethics

**Legal Compliance:**
- Platform terms compliance
- Legal jurisdiction awareness
- Regulatory compliance
- Ethical considerations

**Risk Management:**
- Risk assessment
- Mitigation strategies
- Contingency planning
- Emergency response

## Conclusion

This anti-detection strategy provides a comprehensive approach to placing bets on FanDuel and DraftKings using BrowserBase while minimizing detection risk. The strategy combines advanced technical measures with human-like behavior patterns to create a robust and effective betting automation system.

Key success factors include:
- Consistent behavior randomization
- Platform-specific adaptation
- Regular strategy updates
- Comprehensive monitoring
- Ethical and compliant operation

Regular review and updates of this strategy are essential to maintain effectiveness as platforms evolve their detection methods. 