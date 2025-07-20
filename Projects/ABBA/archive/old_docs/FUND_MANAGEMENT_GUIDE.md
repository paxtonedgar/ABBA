# Fund Management & Balance Monitoring Guide

## Overview

This guide explains how the ABMBA system monitors betting platform balances and manages fund synchronization between you and the agents using BrowserBase for undetected automation.

## 🎯 How Balance Monitoring Works

### **Anti-Detection Balance Checking**

The system uses BrowserBase to check your account balances on FanDuel and DraftKings without detection:

```python
# Example balance check flow
1. Create BrowserBase session with stealth mode
2. Navigate to account/balance page
3. Extract balance using multiple methods
4. Cache results to avoid excessive checking
5. Rotate sessions for anti-detection
```

### **Balance Extraction Methods**

The system uses multiple techniques to extract balances:

1. **CSS Selectors**: Multiple selectors for each platform
   ```python
   balance_selectors = [
       ".balance",
       "[data-testid='balance']", 
       ".account-balance",
       ".user-balance",
       ".wallet-balance"
   ]
   ```

2. **JavaScript Evaluation**: Fallback method using browser JavaScript
   ```javascript
   document.querySelector('.balance')?.textContent
   document.querySelector('[data-testid="balance"]')?.textContent
   ```

3. **Pattern Matching**: Multiple regex patterns to parse balance text
   ```python
   patterns = [
       r'\$?([\d,]+\.?\d*)',  # $1,234.56
       r'([\d,]+\.?\d*)\s*USD',  # 1,234.56 USD
       r'Balance:\s*\$?([\d,]+\.?\d*)'  # Balance: $1,234.56
   ]
   ```

### **Anti-Detection Features**

- **Rate Limiting**: Checks every 5 minutes to avoid excessive requests
- **Session Rotation**: Rotates BrowserBase sessions every hour
- **Human-Like Behavior**: Random delays and natural navigation patterns
- **Stealth Mode**: Full BrowserBase stealth configuration
- **Multiple Fallbacks**: If one method fails, tries others

## 🔄 Fund Management Workflow

### **1. Continuous Monitoring**

The system continuously monitors your balances:

```python
# Monitoring loop
while running:
    for platform in [fanduel, draftkings]:
        balance = check_balance(platform)
        if balance < threshold:
            notify_fund_needed(platform, balance)
    sleep(5_minutes)
```

### **2. Fund Request Process**

When balances are low, the system requests funds:

```python
# Automatic fund request
if balance < auto_request_threshold:
    request_id = request_funds(platform, amount, reason)
    send_notifications(request_id)
```

### **3. Human-Agent Interaction**

You receive notifications and can confirm deposits:

```python
# You make deposit manually on platform
# Then confirm with agent
await system.confirm_deposit("draftkings", 200.00, "confirmation_id")
```

## 📱 Notification System

### **Multi-Channel Notifications**

The system can notify you through multiple channels:

1. **Console**: Real-time console output
2. **Email**: Email notifications
3. **Slack**: Slack webhook messages
4. **SMS**: Text message alerts
5. **Webhook**: Custom webhook endpoints

### **Notification Examples**

```python
# Fund needed notification
🚨 FUND ALERT: DRAFTKINGS
   Current Balance: $15.50
   Status: critical
   Warnings: ['CRITICAL: Balance $15.50 is below critical threshold $20.00']
   Time: 2025-01-19 22:30:15

# Deposit confirmation
✅ DEPOSIT CONFIRMED: DRAFTKINGS received $200.00
```

## 🎛️ Management Modes

### **1. Manual Mode**
- Agent only notifies you of low balances
- You manually request funds when needed
- No automatic fund requests

### **2. Auto Mode**
- Agent automatically requests funds when balances are low
- You receive notifications and confirm deposits
- Fully automated fund management

### **3. Hybrid Mode** (Recommended)
- Agent notifies you AND automatically requests funds
- You can override or modify requests
- Best of both worlds

## 💰 Fund Management Process

### **Step 1: Balance Monitoring**
```python
# System checks balances every 5 minutes
balance_info = await balance_monitor.check_balance("draftkings", credentials)
# Returns: BalanceInfo(current_balance=15.50, status=CRITICAL)
```

### **Step 2: Fund Request**
```python
# System creates fund request
request_id = await interface.request_funds(
    platform="draftkings",
    amount=200.00,
    reason="Auto-request: Balance $15.50 below threshold $50.00"
)
# Returns: "draftkings_1705698015_1234"
```

### **Step 3: Human Action**
```bash
# You receive notification and deposit funds manually
# Go to DraftKings website/app
# Deposit $200
# Note the confirmation ID (optional)
```

### **Step 4: Deposit Confirmation**
```python
# You confirm the deposit with the agent
await system.confirm_deposit("draftkings", 200.00, "confirmation_id")
# System verifies new balance and updates records
```

### **Step 5: Balance Verification**
```python
# System automatically verifies the new balance
new_balance = await balance_monitor.check_balance("draftkings", credentials)
# Updates cache and marks request as confirmed
```

## 🔧 Configuration

### **Environment Variables**
```bash
# BrowserBase Configuration
BROWSERBASE_API_KEY=your_api_key
BROWSERBASE_PROJECT_ID=your_project_id
BROWSERBASE_PROXY=your_proxy_url  # Optional

# Platform Credentials
FANDUEL_USERNAME=your_username
FANDUEL_PASSWORD=your_password
DRAFTKINGS_USERNAME=your_username
DRAFTKINGS_PASSWORD=your_password
```

### **Configuration File (config.yaml)**
```yaml
fund_management:
  mode: "hybrid"  # manual, auto, hybrid
  auto_request_threshold: 50.00  # Request funds when below $50
  request_amount: 200.00  # Request $200 at a time
  minimum_balance: 100.00  # Minimum balance to maintain
  check_interval: 300  # Check every 5 minutes
  notification_channels: ["console", "email", "slack"]
  platforms: ["fanduel", "draftkings"]
```

## 🛡️ Anti-Detection Strategies

### **BrowserBase Stealth Configuration**
```python
stealth_config = {
    "enabled": True,
    "viewport": {"width": random.randint(1820, 2020), "height": random.randint(1030, 1130)},
    "userAgent": random.choice(user_agents),
    "webgl": {"enabled": True, "vendor": "Google Inc. (Intel)"},
    "canvas": {"enabled": True, "noise": True},
    "audio": {"enabled": True, "noise": True},
    "timezone": {"enabled": True, "value": random.choice(timezones)},
    "geolocation": {"enabled": True, "latitude": random.uniform(40.0, 45.0)}
}
```

### **Human-Like Behavior**
- Random delays between actions (1-4 seconds)
- Natural typing patterns (50-150ms between keystrokes)
- Variable mouse movements and click patterns
- Session rotation every hour
- Rate limiting to avoid excessive requests

### **Platform-Specific Adaptations**
- Different selectors for each platform
- Platform-specific timing patterns
- Adaptive behavior based on platform changes
- Graceful error handling and fallbacks

## 📊 Monitoring & Reporting

### **System Status**
```python
status = await system.get_system_status()
# Returns comprehensive system status including:
# - Current balances for all platforms
# - Pending fund requests
# - Recent deposits
# - System configuration
# - Monitoring status
```

### **Balance Summary**
```python
summary = await system.get_balance_summary()
# Returns:
{
    "total_balance": 1250.75,
    "platforms": {
        "fanduel": {"balance": 750.25, "status": "sufficient"},
        "draftkings": {"balance": 500.50, "status": "sufficient"}
    },
    "low_balance_platforms": [],
    "critical_platforms": []
}
```

### **Fund Requests**
```python
requests = await system.get_fund_requests(FundRequestStatus.PENDING)
# Returns pending fund requests with details
```

## 🚀 Usage Examples

### **Basic Usage**
```python
# Initialize system
system = ABMBAFundManagementSystem()
await system.initialize()

# Start monitoring
await system.start_monitoring()

# Check balances
summary = await system.get_balance_summary()

# Request funds manually
request_id = await system.request_funds_manual("draftkings", 200.00, "Manual request")

# Confirm deposit
success = await system.confirm_deposit("draftkings", 200.00, "confirmation_123")

# Get system status
status = await system.get_system_status()

# Close system
await system.close()
```

### **CLI Interface**
```bash
# Run interactive CLI
python fund_management_integration.py

# Options available:
# 1. Start monitoring
# 2. Stop monitoring  
# 3. Check all balances
# 4. Check specific platform balance
# 5. Request funds manually
# 6. Confirm deposit
# 7. View pending requests
# 8. Get system status
# 9. Get fund management report
# 10. Exit
```

## 🔍 Troubleshooting

### **Common Issues**

1. **401 Unauthorized Error**
   - Check BrowserBase API credentials
   - Verify project ID is correct

2. **Balance Extraction Failed**
   - Platform may have changed selectors
   - Check for platform updates
   - System will try multiple fallback methods

3. **Login Failed**
   - Check platform credentials
   - Platform may have 2FA enabled
   - Check for account restrictions

4. **Session Creation Failed**
   - Check BrowserBase API key
   - Verify project configuration
   - Check rate limits

### **Debug Mode**
```python
import structlog
structlog.configure(processors=[structlog.processors.JSONRenderer()])
# Enable detailed logging for troubleshooting
```

## 📈 Best Practices

### **Security**
- Use environment variables for credentials
- Rotate credentials regularly
- Monitor for suspicious activity
- Use secure notification channels

### **Performance**
- Don't check balances too frequently
- Use caching to reduce API calls
- Monitor BrowserBase usage costs
- Optimize session management

### **Reliability**
- Implement proper error handling
- Use multiple fallback methods
- Monitor system health
- Regular testing and validation

### **Compliance**
- Follow platform terms of service
- Respect rate limits
- Monitor for policy changes
- Implement responsible betting practices

## 🎯 Key Benefits

1. **Undetected Operation**: BrowserBase stealth mode prevents detection
2. **Real-Time Monitoring**: Continuous balance tracking
3. **Automated Management**: Reduces manual intervention
4. **Multi-Platform Support**: Works with FanDuel and DraftKings
5. **Flexible Configuration**: Multiple management modes
6. **Comprehensive Notifications**: Multiple notification channels
7. **Robust Error Handling**: Graceful degradation and recovery
8. **Detailed Reporting**: Complete system status and history

## 🔮 Future Enhancements

1. **Additional Platforms**: Support for more betting platforms
2. **Advanced Analytics**: Predictive balance management
3. **Mobile Integration**: Mobile app automation
4. **AI-Powered Decisions**: Machine learning for fund management
5. **Blockchain Integration**: Cryptocurrency deposit support
6. **Advanced Notifications**: Push notifications, voice alerts
7. **Portfolio Management**: Multi-account management
8. **Risk Assessment**: Advanced risk management features

---

This fund management system provides a comprehensive, undetected solution for managing betting platform balances while maintaining synchronization between you and your agents. The BrowserBase integration ensures that all balance checking and fund management operations are performed without detection, allowing your agents to operate effectively while you maintain control over your funds. 