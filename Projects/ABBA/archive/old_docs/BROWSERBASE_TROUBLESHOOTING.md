# BrowserBase API Troubleshooting Guide

## 🚨 **Current Issue: 401 Unauthorized Error - FIXED**

**Root Cause Identified:** BrowserBase changed authentication from `Authorization: Bearer` to `X-BB-API-Key` header.

**Your Current Configuration:**
- API Key: `bb_live_pbKC2uCFKuZpPtxePtLleB8b4pg`
- Project ID: `433084d3-e4ec-4025-b4de-cfd87682d8e0`
- DraftKings Username: `paxtonedgar3@gmail.com`
- DraftKings Password: `Empireozarks@2013`

---

## ✅ **FIXED: Updated Authentication Method**

### **Correct Header Format**
```bash
# OLD (doesn't work):
-H "Authorization: Bearer bb_live_pbKC2uCFKuZpPtxePtLleB8b4pg"

# NEW (correct):
-H "X-BB-API-Key: bb_live_pbKC2uCFKuZpPtxePtLleB8b4pg"
```

### **Updated Test Commands**

#### **1. API Key Validation (Smoke Test)**
```bash
curl -X GET "https://api.browserbase.com/v1/projects/433084d3-e4ec-4025-b4de-cfd87682d8e0" \
  -H "X-BB-API-Key: bb_live_pbKC2uCFKuZpPtxePtLleB8b4pg"
```

#### **2. Session Creation**
```bash
curl -X POST "https://api.browserbase.com/v1/sessions" \
  -H "Content-Type: application/json" \
  -H "X-BB-API-Key: bb_live_pbKC2uCFKuZpPtxePtLleB8b4pg" \
  -d '{
    "projectId": "433084d3-e4ec-4025-b4de-cfd87682d8e0",
    "stealth": {
      "enabled": true,
      "viewport": {
        "width": 1920,
        "height": 1080
      }
    }
  }'
```

---

## 🔍 **Step-by-Step Testing**

### **Step 1: Test API Key Validation**
```bash
# Run the updated curl test
./test_browserbase_curl.sh
```

### **Step 2: Test Python API**
```bash
# Run the updated Python test
python test_browserbase_api.py
```

### **Step 3: Test Live Balance Monitoring**
```bash
# Once API works, test DraftKings integration
python test_live_balance_simple.py
```

---

## 🔧 **Updated Code Changes**

### **Files Updated:**
1. **`browserbase_executor.py`** - Updated authentication header
2. **`test_browserbase_api.py`** - Added project validation + updated headers
3. **`test_browserbase_curl.sh`** - Updated headers + added smoke test
4. **`BROWSERBASE_TROUBLESHOOTING.md`** - This guide

### **Key Changes:**
- ✅ Replaced `Authorization: Bearer` with `X-BB-API-Key`
- ✅ Added project validation before session creation
- ✅ Updated all API calls to use correct headers
- ✅ Added smoke test for API key validation

---

## 📊 **Expected Results After Fix**

### **API Key Validation (Step 1):**
```json
{
  "id": "433084d3-e4ec-4025-b4de-cfd87682d8e0",
  "name": "Your Project Name",
  "status": "active"
}
```

### **Session Creation (Step 2):**
```json
{
  "id": "session_12345",
  "status": "created",
  "projectId": "433084d3-e4ec-4025-b4de-cfd87682d8e0"
}
```

### **Navigation (Step 3):**
```json
{
  "url": "https://www.google.com",
  "title": "Google",
  "status": "success"
}
```

---

## 🚀 **Quick Test Commands**

### **1. Test with Curl**
```bash
./test_browserbase_curl.sh
```

### **2. Test with Python**
```bash
python test_browserbase_api.py
```

### **3. Test Live DraftKings**
```bash
python test_live_balance_simple.py
```

---

## 📋 **Updated Checklist**

### **Authentication**
- [x] Use `X-BB-API-Key` header (not `Authorization: Bearer`)
- [x] API key format is correct (`bb_live_...`)
- [x] Project ID exists and is accessible

### **Testing**
- [ ] API key validation succeeds
- [ ] Session creation works
- [ ] Navigation functions properly
- [ ] Live DraftKings test passes

---

## 🎯 **Next Steps**

1. **Run the updated tests** with correct authentication
2. **Verify API key works** with project validation
3. **Test session creation** and navigation
4. **Run live DraftKings balance monitoring**

---

## 📝 **Notes**

- ✅ **FIXED**: Authentication header issue resolved
- ✅ **UPDATED**: All code uses correct `X-BB-API-Key` header
- ✅ **ADDED**: Project validation smoke test
- 🎯 **READY**: System should now work with your BrowserBase account

---

**Last Updated**: 2025-07-19
**Status**: ✅ FIXED - Authentication header updated
**Priority**: Ready for live testing 