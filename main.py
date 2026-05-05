import argparse
from fdm_engine.config import logger, OptionParameters, GridParameters

def main():
    parser = argparse.ArgumentParser(description="High-Performance American Option Pricing Engine")
    parser.add_argument("--s0", type=float, default=100.0, help="Initial asset price")
    parser.add_argument("--k", type=float, default=100.0, help="Strike price")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--t", type=float, default=1.0, help="Time to maturity")
    
    args = parser.parse_args()
    
    logger.info("Initializing Option Pricing Engine...")
    
    params = OptionParameters(
        S0=args.s0,
        K=args.k,
        r=args.r,
        sigma=args.sigma,
        T=args.t
    )
    
    grid_params = GridParameters()
    
    logger.info(f"Parameters: {params}")
    logger.info(f"Grid Parameters: {grid_params}")

if __name__ == "__main__":
    main()
