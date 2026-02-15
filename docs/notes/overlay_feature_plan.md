# Production MTGA Draft Overlay Feature Plan

## 1. Core Overlay UI

### 1.1 Card List with Scores
- Sorted by predicted net effect (or state value).
- grpId → card name mapping.
- Score formatting and visual emphasis.

### 1.2 Color-Coded Strength Tiers
- A = top 5% predicted deck improvement  
- B = 5%–20%  
- C = 20%–50%  
- D = 50%–80%  
- F = bottom 20%

### 1.3 Current Pick Context
- Pack X / Pick Y  
- Pool color summary  
- Curve summary  

## 2. Card View & Imagery

### 2.1 Card Image Preview on Hover
- Fetch from ArtId-derived URL.

### 2.2 Tooltip Info
- Mana cost  
- Types  
- Rarity  
- Power/Toughness  
- Text  
- Synergies  
- Signals  

## 3. Advanced Draft Guidance Features

### 3.1 Real-Time Synergy Score
`s(state + card) – s(state)`

### 3.2 Color Commitment Score
Shows color leaning and suggested pair.

### 3.3 Fixing Warnings
Warns when the player is overreaching on colors.

### 3.4 Curve Visualization
Mini chart showing mana curve distribution.

### 3.5 Draft Signals
Shows openness of colors based on passed cards.

## 4. User-Controlled Settings
- Toggle images  
- Toggle decimals  
- Toggle synergy  
- Transparency slider  
- Overlay position  
- Click-through mode  

## 5. On-Top & Click-Through Behavior
- Use Tauri/Electron/PyQt wrapper  
- Always on top  
- Frameless  
- Click-through  

## 6. Communication Model

```
MTGA Player.log 
   ↓
ArenaScanner
   ↓
FastAPI /state
   ↓
Overlay (Tauri/Electron)
   ↓
UI
```

## 7. Stretch Goals

### 7.1 Draft Replay Mode
Record packs, picks, scores.

### 7.2 AI Auto-Draft
Bot drafts automatically.

### 7.3 Coaching Mode
Explain picks using deltas in expected value.

### 7.4 Meta Comparison
Compare to 17Lands user pick rates.

## 8. Next Steps Options

### (A) Enhanced Overlay UI
### (B) Desktop Overlay Wrapper
### (C) Synergy + Color Commitment Integration
### (D) Full DraftHelper V1 Packaging
