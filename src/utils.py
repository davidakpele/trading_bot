# src/utils.py
import MetaTrader5 as mt5
import time
import logging
from decimal import Decimal
import pandas as pd

logger = logging.getLogger("scalping")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# ============================================================================
# CONNECTION FUNCTIONS
# ============================================================================

def connect_mt5(login=None, password=None, server=None, path=None):
    """
    Connect to MT5 terminal. If login/password provided, use mt5.login.
    """
    if path:
        if not mt5.initialize(path):
            logger.error(f"MT5 initialization failed with path: {path}")
            return False
    else:
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
            
    if login:
        ok = mt5.login(login, password=password, server=server)
        if not ok:
            logger.error(f"MT5 login failed. Error: {mt5.last_error()}")
            return False
    logger.info("MT5 initialized successfully")
    return True

def disconnect_mt5():
    try:
        mt5.shutdown()
        logger.info("MT5 disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting MT5: {e}")

# ============================================================================
# DATA RETRIEVAL FUNCTIONS
# ============================================================================

def get_latest_ticks(symbol, n=100):
    """Return the last n minute bars using mt5.copy_rates_from_pos"""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n)
    if rates is None:
        logger.error(f"Failed to get rates for {symbol}. Error: {mt5.last_error()}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'time':'timestamp'})
    return df[['timestamp','open','high','low','close','tick_volume']].rename(columns={'tick_volume':'volume'})

def get_account_info():
    """Get current account information"""
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info")
        return None
    return account_info

def get_open_positions(symbol=None):
    """Get open positions, optionally filtered by symbol"""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if positions is None:
        return []
    return positions

# ============================================================================
# IMPROVED ORDER EXECUTION FUNCTIONS
# ============================================================================

def place_order_market_improved(symbol, order_type, lots=0.01, sl_pips=8, tp_pips=12, deviation=50, max_retries=5):
    """
    Improved market execution with better retry logic and filling policy
    """
    for attempt in range(max_retries):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
            
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Could not get tick for {symbol}")
            time.sleep(0.1)
            continue
        
        digits = symbol_info.digits
        pip_size = 0.0001 if digits == 5 else 0.01 if digits == 3 else 0.001
        if order_type == 'buy':
            price = tick.ask
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            sl = round(price - sl_pips * pip_size, digits) if sl_pips else 0.0
            tp = round(price + tp_pips * pip_size, digits) if tp_pips else 0.0
        else: 
            price = tick.bid
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            sl = round(price + sl_pips * pip_size, digits) if sl_pips else 0.0
            tp = round(price - tp_pips * pip_size, digits) if tp_pips else 0.0
        current_deviation = deviation + (attempt * 10)
        if attempt < 2:
            filling_type = mt5.ORDER_FILLING_FOK 
        else:
            filling_type = mt5.ORDER_FILLING_RETURN  
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type_mt5,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": current_deviation,
            "magic": 123456,
            "comment": f"scalping-bot-v2-{attempt+1}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"order_send returned None on attempt {attempt+1}")
            time.sleep(0.2)
            continue
        
        # Handle result
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✓ Order executed successfully on attempt {attempt+1}")
            logger.info(f"  Requested: {price:.5f}, Executed: {result.price:.5f}, Volume: {result.volume}")
            return result
            
        elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:  
            logger.warning(f"Requote on attempt {attempt+1}/{max_retries}")
            logger.info(f"  Requested: {price:.5f}, Deviation: {current_deviation}")
            time.sleep(0.3)
            continue
            
        elif result.retcode == mt5.TRADE_RETCODE_PRICE_OFF:  
            logger.warning(f"Invalid price on attempt {attempt+1}, refreshing...")
            time.sleep(0.2)
            continue
            
        elif result.retcode == mt5.TRADE_RETCODE_REJECT:  
            logger.error(f"Order rejected by broker: {result.comment}")
            return result
            
        elif result.retcode == mt5.TRADE_RETCODE_INVALID_FILL: 
            logger.warning(f"Invalid filling mode, switching to RETURN")
            time.sleep(0.1)
            continue
            
        else:
            logger.error(f"Order failed with code {result.retcode}: {result.comment}")
            if attempt == max_retries - 1:
                return result
            time.sleep(0.2)
            continue
    
    logger.error(f"Order failed after {max_retries} attempts")
    return result


def place_order_with_slippage_check(symbol, order_type, lots=0.01, sl_pips=8, tp_pips=12, 
                                     max_slippage_pips=5, max_retries=5):
    """
    Place order with slippage monitoring - rejects if slippage exceeds threshold
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbol {symbol} not found")
        return None
    
    digits = symbol_info.digits
    pip_size = 0.0001 if digits == 5 else 0.01 if digits == 3 else 0.001
    
    for attempt in range(max_retries):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Could not get tick for {symbol}")
            time.sleep(0.1)
            continue
        
        initial_price = tick.ask if order_type == 'buy' else tick.bid
        result = place_order_market_improved(symbol, order_type, lots, sl_pips, tp_pips, 
                                             deviation=100, max_retries=1)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            executed_price = result.price
            slippage = abs(executed_price - initial_price) / pip_size
            if slippage > max_slippage_pips:
                logger.warning(f"Slippage too high: {slippage:.2f} pips (max: {max_slippage_pips})")
                logger.warning(f"  Initial: {initial_price:.5f}, Executed: {executed_price:.5f}")
                positions = mt5.positions_get(symbol=symbol)
                if positions and len(positions) > 0:
                    close_position(positions[-1])
                    logger.info("Position closed due to excessive slippage")
                
                time.sleep(0.5)
                continue
            else:
                logger.info(f"✓ Order executed with acceptable slippage: {slippage:.2f} pips")
                return result
        else:
            time.sleep(0.3)
            continue
    
    logger.error(f"Failed to execute order with acceptable slippage after {max_retries} attempts")
    return None


def close_position(position, deviation=50, max_retries=3):
    """
    Close a specific position with retry logic
    """
    for attempt in range(max_retries):
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            logger.error(f"Could not get tick for {position.symbol}")
            time.sleep(0.1)
            continue
        
        if position.type == mt5.ORDER_TYPE_BUY:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        
        current_deviation = deviation + (attempt * 10)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": current_deviation,
            "magic": 123456,
            "comment": f"close-v2-{attempt+1}",
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✓ Position closed successfully on attempt {attempt+1}")
            return result
        elif result:
            logger.warning(f"Close failed on attempt {attempt+1}: {result.retcode} - {result.comment}")
            time.sleep(0.2)
        else:
            logger.error(f"Close order_send returned None on attempt {attempt+1}")
            time.sleep(0.2)
    
    logger.error(f"Failed to close position after {max_retries} attempts")
    return result


# ============================================================================
# LEGACY FUNCTIONS (for backward compatibility)
# ============================================================================

def place_order(symbol, order_type, lots=0.01, price=None, sl=None, tp=None, deviation=20, max_retries=3):
    """
    Legacy place_order function - redirects to improved version
    Convert pip-based sl/tp to price-based if needed
    """
    logger.warning("Using legacy place_order - consider switching to place_order_market_improved")
    
    if sl or tp:
        symbol_info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if symbol_info and tick:
            digits = symbol_info.digits
            pip_size = 0.0001 if digits == 5 else 0.01 if digits == 3 else 0.001
            current_price = tick.ask if order_type == 'buy' else tick.bid
            
            sl_pips = abs(current_price - sl) / pip_size if sl else 8
            tp_pips = abs(tp - current_price) / pip_size if tp else 12
            
            return place_order_market_improved(symbol, order_type, lots, sl_pips, tp_pips, deviation, max_retries)
    return place_order_market_improved(symbol, order_type, lots, 8, 12, deviation, max_retries)


def place_order_market(symbol, order_type, lots=0.01, sl_pips=8, tp_pips=12, deviation=20):
    """
    Legacy market execution - redirects to improved version
    """
    logger.warning("Using legacy place_order_market - consider switching to place_order_market_improved")
    return place_order_market_improved(symbol, order_type, lots, sl_pips, tp_pips, deviation, max_retries=5)


def add_stop_loss_to_position(position_ticket, sl_pips=10):
    """
    Add stop loss to an existing position that doesn't have one
    """
    positions = mt5.positions_get(ticket=position_ticket)
    if not positions:
        logger.error(f"Position {position_ticket} not found")
        return None
    
    position = positions[0]
    symbol_info = mt5.symbol_info(position.symbol)
    if not symbol_info:
        logger.error(f"Could not get symbol info for {position.symbol}")
        return None
    
    digits = symbol_info.digits
    pip_size = 0.0001 if digits == 5 else 0.01 if digits == 3 else 0.001
    if position.type == mt5.ORDER_TYPE_SELL:  
        sl_price = position.price_open + sl_pips * pip_size
    else: 
        sl_price = position.price_open - sl_pips * pip_size
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position_ticket,
        "sl": float(sl_price),
        "magic": 123456,
        "comment": "added-stop-loss",
    }
    
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"Stop loss added to position {position_ticket}: SL={sl_price:.5f}")
    elif result:
        logger.error(f"Failed to add stop loss: {result.retcode} - {result.comment}")
    
    return result


def add_stop_loss_to_all_positions(sl_pips=10):
    """
    Add stop loss to ALL open positions that don't have one
    """
    positions = mt5.positions_get()
    if not positions:
        logger.info("No open positions found")
        return
    
    fixed_count = 0
    for position in positions:
        if position.sl == 0.0: 
            result = add_stop_loss_to_position(position.ticket, sl_pips)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                fixed_count += 1
    
    logger.info(f"Added stop loss to {fixed_count} positions")
    
    

def check_and_fix_positions():
    """
    Check all positions and fix any missing stop losses
    """
    positions = mt5.positions_get()
    if not positions:
        return
    
    risky_positions = []
    for position in positions:
        if position.sl == 0.0: 
            risky_positions.append({
                'ticket': position.ticket,
                'symbol': position.symbol,
                'type': 'SELL' if position.type == mt5.ORDER_TYPE_SELL else 'BUY',
                'volume': position.volume,
                'open_price': position.price_open,
                'current_price': position.price_current,
                'profit': position.profit
            })

    if risky_positions:
        logger.warning(f"Found {len(risky_positions)} positions without stop loss:")
        for pos in risky_positions:
            logger.warning(f"   {pos['symbol']} {pos['type']} (Ticket: {pos['ticket']}, Profit: ${pos['profit']:.2f})")
        add_stop_loss_to_all_positions(sl_pips=10)
    else:
        logger.info("All positions have proper stop loss protection")
