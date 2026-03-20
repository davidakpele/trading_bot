# src/check_stoploss.py
from src.utils import connect_mt5, disconnect_mt5, get_open_positions, add_stop_loss_to_all_positions, check_and_fix_positions

def check_stop_loss_status():
    """Check if stop losses are properly set on all positions"""
    
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
    
    print("CHECKING STOP LOSS STATUS")
    print("=" * 60)
    
    # Get all positions
    positions = get_open_positions()
    
    if not positions:
        print("No open positions found")
        disconnect_mt5()
        return
    
    print(f"Found {len(positions)} open positions:")
    print("-" * 60)
    
    risky_count = 0
    protected_count = 0
    
    for pos in positions:
        symbol_info = mt5.symbol_info(pos.symbol)
        digits = symbol_info.digits if symbol_info else 5
        pip_size = 0.0001 if digits == 5 else 0.01 if digits == 3 else 0.001
        
        # Calculate distance to SL in pips
        if pos.sl > 0:
            if pos.type == mt5.ORDER_TYPE_SELL:  
                sl_distance = (pos.sl - pos.price_current) / pip_size
            else: 
                sl_distance = (pos.price_current - pos.sl) / pip_size
            
            status = "PROTECTED"
            protected_count += 1
        else:
            sl_distance = 0
            status = "NO STOP LOSS"
            risky_count += 1
        
        print(f"Symbol: {pos.symbol}")
        print(f"Type:   {'SELL' if pos.type == mt5.ORDER_TYPE_SELL else 'BUY'}")
        print(f"Open:   {pos.price_open:.5f}")
        print(f"Current: {pos.price_current:.5f}")
        print(f"SL:     {pos.sl:.5f}" if pos.sl > 0 else "SL:  NONE")
        print(f"TP:     {pos.tp:.5f}" if pos.tp > 0 else "TP:  NONE")
        print(f"Status: {status}")
        if pos.sl > 0:
            print(f"SL Distance: {sl_distance:.1f} pips")
        print(f"Profit: ${pos.profit:.2f}")
        print("-" * 40)
    
    # Summary
    print("\nSUMMARY:")
    print(f"Protected positions: {protected_count}")
    print(f"Risky positions: {risky_count} ")
    
    if risky_count > 0:
        print(f"\n EMERGENCY: {risky_count} positions without stop loss!")
        print("Fixing automatically...")
        add_stop_loss_to_all_positions(sl_pips=10)
        
        # Verify fix
        print("\nVERIFYING FIX...")
        positions_after = get_open_positions()
        still_risky = sum(1 for p in positions_after if p.sl == 0.0)
        
        if still_risky == 0:
            print(" SUCCESS: All positions now have stop loss protection!")
        else:
            print(f" FAILED: {still_risky} positions still without stop loss")
    else:
        print("All positions are properly protected with stop loss!")
    
    disconnect_mt5()

def test_stop_loss_calculation():
    """Test if stop loss calculations are correct"""
    print("\nTESTING STOP LOSS CALCULATIONS")
    print("=" * 50)
    
    test_cases = [
        {"symbol": "EURUSD", "type": "SELL", "open_price": 1.15822, "sl_pips": 10},
        {"symbol": "EURUSD", "type": "BUY", "open_price": 1.15822, "sl_pips": 10},
        {"symbol": "USDJPY", "type": "SELL", "open_price": 155.400, "sl_pips": 10},
        {"symbol": "USDJPY", "type": "BUY", "open_price": 155.400, "sl_pips": 10},
    ]
    
    for test in test_cases:
        symbol = test["symbol"]
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            digits = symbol_info.digits
            pip_size = 0.0001 if digits == 5 else 0.01 if digits == 3 else 0.001
            
            if test["type"] == "SELL":
                sl_price = test["open_price"] + test["sl_pips"] * pip_size
            else:  # BUY
                sl_price = test["open_price"] - test["sl_pips"] * pip_size
            
            print(f"{symbol} {test['type']}:")
            print(f"  Open: {test['open_price']:.5f}")
            print(f"  SL ({test['sl_pips']} pips): {sl_price:.5f}")
            print(f"  Distance: {test['sl_pips']} pips")
            print()

if __name__ == "__main__":
    # Run the diagnostic
    check_stop_loss_status()
    
    # Run calculation test
    if connect_mt5():
        test_stop_loss_calculation()
        disconnect_mt5()
