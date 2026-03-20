# src/monitor.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
import os
import time

class TradingMonitor:
    """
    Real-time monitor for MT5 trading bot
    Shows positions, history, and performance metrics
    """
    
    def __init__(self, symbol=None):
        self.symbol = symbol
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_account_summary(self):
        """Get account balance and equity info"""
        account = mt5.account_info()
        if account is None:
            return None
        
        return {
            'Balance': f"${account.balance:.2f}",
            'Equity': f"${account.equity:.2f}",
            'Margin': f"${account.margin:.2f}",
            'Free Margin': f"${account.margin_free:.2f}",
            'Profit': f"${account.profit:.2f}",
            'Margin Level': f"{account.margin_level:.2f}%" if account.margin > 0 else "N/A"
        }
    
    def get_open_positions(self):
        """Get all open positions"""
        positions = mt5.positions_get(symbol=self.symbol) if self.symbol else mt5.positions_get()
        
        if positions is None or len(positions) == 0:
            return pd.DataFrame()
        
        pos_list = []
        for pos in positions:
            # Calculate current P/L
            current_profit = pos.profit
            
            # Calculate pips
            symbol_info = mt5.symbol_info(pos.symbol)
            if symbol_info:
                pip_size = 0.0001 if symbol_info.digits == 5 else 0.01 if symbol_info.digits == 3 else 0.001
                pips = (pos.price_current - pos.price_open) / pip_size
                if pos.type == mt5.ORDER_TYPE_SELL:
                    pips = -pips
            else:
                pips = 0
            
            pos_list.append({
                'Ticket': pos.ticket,
                'Symbol': pos.symbol,
                'Type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'Volume': pos.volume,
                'Open Price': f"{pos.price_open:.5f}",
                'Current': f"{pos.price_current:.5f}",
                'SL': f"{pos.sl:.5f}" if pos.sl > 0 else "None",
                'TP': f"{pos.tp:.5f}" if pos.tp > 0 else "None",
                'Pips': f"{pips:.1f}",
                'Profit': f"${current_profit:.2f}",
                'Time': datetime.fromtimestamp(pos.time).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return pd.DataFrame(pos_list)
    
    def get_today_history(self):
        """Get trading history for today"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get deals (actual executions)
        deals = mt5.history_deals_get(today, datetime.now())
        
        if deals is None or len(deals) == 0:
            return pd.DataFrame()
        
        deal_list = []
        for deal in deals:
            if deal.entry == 2:
                deal_list.append({
                    'Time': datetime.fromtimestamp(deal.time).strftime('%H:%M:%S'),
                    'Ticket': deal.ticket,
                    'Symbol': deal.symbol,
                    'Type': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                    'Volume': deal.volume,
                    'Price': f"{deal.price:.5f}",
                    'Profit': f"${deal.profit:.2f}",
                    'Comment': deal.comment[:20] if deal.comment else ""
                })
        
        df = pd.DataFrame(deal_list)
        return df.tail(20) if not df.empty else df 
    
    def get_performance_stats(self):
        """Calculate today's performance statistics"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(today, datetime.now())
        
        if deals is None or len(deals) == 0:
            return None
        
        closed_deals = [d for d in deals if d.entry == 2]  
        
        if len(closed_deals) == 0:
            return None
        
        profits = [d.profit for d in closed_deals]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        total_profit = sum(profits)
        total_trades = len(profits)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        return {
            'Total Trades': total_trades,
            'Winning': winning_trades,
            'Losing': losing_trades,
            'Win Rate': f"{win_rate:.1f}%",
            'Total P/L': f"${total_profit:.2f}",
            'Avg Win': f"${avg_win:.2f}",
            'Avg Loss': f"${avg_loss:.2f}",
            'Profit Factor': f"{profit_factor:.2f}" if profit_factor > 0 else "N/A",
            'Best Trade': f"${max(profits):.2f}",
            'Worst Trade': f"${min(profits):.2f}"
        }
    
    def display_dashboard(self):
        """Display the complete monitoring dashboard"""
        self.clear_screen()
        
        print("=" * 100)
        print(f"🤖 SCALPING BOT MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
        print()
        
        # Account Summary
        print("ACCOUNT SUMMARY")
        print("-" * 100)
        account = self.get_account_summary()
        if account:
            for key, value in account.items():
                print(f"{key:15}: {value}")
        print()
        
        # Open Positions
        print("OPEN POSITIONS")
        print("-" * 100)
        positions = self.get_open_positions()
        if not positions.empty:
            print(tabulate(positions, headers='keys', tablefmt='grid', showindex=False))
        else:
            print("No open positions")
        print()
        
        # Today's Performance
        print("TODAY'S PERFORMANCE")
        print("-" * 100)
        stats = self.get_performance_stats()
        if stats:
            for key, value in stats.items():
                print(f"{key:15}: {value}")
        else:
            print("No closed trades today")
        print()
        
        # Recent History
        print("RECENT TRADE HISTORY (Last 20)")
        print("-" * 100)
        history = self.get_today_history()
        if not history.empty:
            print(tabulate(history, headers='keys', tablefmt='grid', showindex=False))
        else:
            print("No trade history today")
        print()
        
        print("=" * 100)
        print("Press Ctrl+C to stop monitoring")
        print("=" * 100)
    
    def run_monitor(self, refresh_interval=5):
        """
        Run the monitor in a loop
        
        Args:
            refresh_interval: Seconds between updates (default: 5)
        """
        print("Starting Trading Monitor...")
        print(f"Monitoring symbol: {self.symbol if self.symbol else 'ALL'}")
        print(f"Refresh interval: {refresh_interval} seconds")
        print()
        time.sleep(2)
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user")


def run_standalone_monitor(symbol=None, refresh_interval=5):
    """
    Run the monitor as a standalone application
    
    Usage:
        python monitor.py
        python monitor.py --symbol EURUSD
        python monitor.py --symbol EURUSD --interval 10
    """
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    
    print(f"MT5 initialized successfully")
    
    monitor = TradingMonitor(symbol=symbol)
    monitor.run_monitor(refresh_interval=refresh_interval)
    
    mt5.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MT5 Trading Bot Monitor')
    parser.add_argument('--symbol', type=str, default=None, help='Symbol to monitor (e.g., EURUSD)')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds')
    parser.add_argument('--mt5-path', type=str, default=None, help='Path to MT5 terminal')
    
    args = parser.parse_args()
    
    # Initialize MT5
    if args.mt5_path:
        if not mt5.initialize(args.mt5_path):
            print(f"Failed to initialize MT5 with path: {args.mt5_path}")
            exit(1)
    else:
        if not mt5.initialize():
            print("Failed to initialize MT5")
            exit(1)
    
    print(f"MT5 initialized successfully")
    
    # Run monitor
    monitor = TradingMonitor(symbol=args.symbol)
    monitor.run_monitor(refresh_interval=args.interval)
    
    mt5.shutdown()
