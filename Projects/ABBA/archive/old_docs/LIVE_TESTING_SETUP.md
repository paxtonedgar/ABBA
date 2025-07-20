# Live Testing Setup Guide

## 🚨 **Important Security Notice**

Before testing with your live DraftKings account, please understand:

1. **This will log into your actual DraftKings account**
2. **Use a test account if possible, not your main account**
3. **The system only reads balance information, doesn't place bets**
4. **All operations are logged for transparency**

## 📋 **Prerequisites**

### **1. BrowserBase Account**
- Sign up at https://browserbase.com
- Get your API key and project ID
- BrowserBase has a free tier (limited usage)

### **2. DraftKings Account**
- Active DraftKings account
- Username and password
- Some funds in the account (for testing)

### **3. Environment Setup**
- Python 3.8+ installed
- All dependencies installed (`httpx`, `structlog`, etc.)

## 🔧 **Setup Steps**

### **Step 1: Install Dependencies**
```bash
# Activate your virtual environment
source venv/bin/activate

# Install required packages
pip install httpx structlog pyyaml
```

### **Step 2: Set Environment Variables**
```bash
# BrowserBase Configuration
export BROWSERBASE_API_KEY="your_browserbase_api_key"
export BROWSERBASE_PROJECT_ID="your_browserbase_project_id"

# DraftKings Credentials
export DRAFTKINGS_USERNAME="your_draftkings_username"
export DRAFTKINGS_PASSWORD="your_draftkings_password"
```

### **Step 3: Verify Setup**
```bash
# Check if environment variables are set
echo "BrowserBase API Key: ${BROWSERBASE_API_KEY:0:10}..."
echo "BrowserBase Project ID: ${BROWSERBASE_PROJECT_ID}"
echo "DraftKings Username: ${DRAFTKINGS_USERNAME}"
echo "DraftKings Password: ${DRAFTKINGS_PASSWORD:0:5}..."
```

## 🧪 **Running the Live Test**

### **Option 1: Quick Test**
```bash
# Run the live balance test
python test_live_balance.py
```

### **Option 2: Full System Test**
```bash
# Run the complete fund management system
python fund_management_integration.py
```

### **Option 3: Balance Monitor Only**
```bash
# Run just the balance monitor
python balance_monitor.py
```

## 📊 **What the Test Will Do**

### **1. Balance Check**
- Log into your DraftKings account using BrowserBase
- Navigate to the account/balance page
- Extract your current balance
- Display the balance and status

### **2. Fund Request Test**
- Create a test fund request (doesn't actually request money)
- Test the notification system
- Verify fund request functionality

### **3. Balance Summary**
- Get comprehensive balance information
- Check for pending requests
- Display system status

## 🔍 **Expected Output**

```
🧪 Live Balance Monitoring Test
==================================================
This will test the balance monitoring system with your DraftKings account.
Make sure you have set up your environment variables first.

Do you want to proceed with live testing? (yes/no): yes

✅ All required environment variables found
✅ Test environment setup completed

==================================================
RUNNING TESTS
==================================================

🔍 Testing DraftKings balance checking...
✅ Balance check successful!
   Platform: draftkings
   Current Balance: $125.50
   Status: sufficient
   Last Updated: 2025-01-19 22:45:30

💰 Testing fund request functionality...
✅ Fund request created: draftkings_1705698030_5678
   Pending requests: 1
   - draftkings_1705698030_5678: $100.00 (pending)

📊 Testing balance summary...
✅ Balance summary retrieved:
   Platform: draftkings
   Balance: $125.50
   Status: sufficient
   Pending Requests: 1

==================================================
TEST RESULTS
==================================================
✅ Balance checking: PASSED
✅ Fund request: PASSED
✅ Balance summary: PASSED

🎉 Live testing completed!

📈 Your DraftKings balance: $125.50
```

## ⚠️ **Troubleshooting**

### **Common Issues**

1. **401 Unauthorized Error**
   ```
   ❌ Error testing DraftKings balance: Client error '401 Unauthorized'
   ```
   - Check your BrowserBase API key and project ID
   - Verify your BrowserBase account is active

2. **Login Failed**
   ```
   ❌ Balance check failed
   ```
   - Check your DraftKings username and password
   - Account might have 2FA enabled
   - Account might be restricted

3. **Balance Extraction Failed**
   ```
   ❌ Could not extract balance from draftkings
   ```
   - DraftKings may have changed their website
   - Try again in a few minutes
   - Check for platform updates

4. **Missing Environment Variables**
   ```
   ❌ Missing environment variables: ['BROWSERBASE_API_KEY']
   ```
   - Set all required environment variables
   - Check your shell configuration

### **Debug Mode**
```bash
# Enable detailed logging
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -c "
import structlog
structlog.configure(processors=[structlog.processors.JSONRenderer()])
import test_live_balance
import asyncio
asyncio.run(test_live_balance.main())
"
```

## 🔒 **Security Best Practices**

### **1. Use Environment Variables**
- Never hardcode credentials in scripts
- Use `.env` files for local development
- Use secure environment variable management in production

### **2. Test Account**
- Use a separate DraftKings account for testing
- Don't use your main account with significant funds
- Consider creating a test account with minimal funds

### **3. Monitor Usage**
- Check BrowserBase usage dashboard
- Monitor for unusual activity
- Set up alerts for high usage

### **4. Regular Testing**
- Test the system regularly
- Update selectors if platforms change
- Keep dependencies updated

## 📈 **Next Steps After Testing**

### **1. If Test Passes**
- Configure the full fund management system
- Set up notifications (email, Slack, etc.)
- Adjust thresholds and amounts
- Start continuous monitoring

### **2. If Test Fails**
- Check the troubleshooting section
- Verify all credentials and settings
- Test with BrowserBase playground first
- Contact support if issues persist

### **3. Production Setup**
- Use production BrowserBase account
- Set up proper logging and monitoring
- Configure backup and recovery procedures
- Implement proper error handling

## 🎯 **Success Criteria**

The test is successful if:

1. ✅ **Balance Check**: Successfully retrieves your DraftKings balance
2. ✅ **Fund Request**: Creates and manages fund requests
3. ✅ **Balance Summary**: Provides comprehensive balance information
4. ✅ **No Errors**: All operations complete without errors
5. ✅ **Anti-Detection**: Operations complete without triggering security measures

## 🚀 **Ready to Test?**

If you have:
- ✅ BrowserBase account and credentials
- ✅ DraftKings account credentials
- ✅ Environment variables set
- ✅ Dependencies installed

Then you're ready to run:

```bash
python test_live_balance.py
```

The test will guide you through the process and provide detailed feedback on each step. 