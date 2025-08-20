import warnings
import sys
import asyncio

from logger_config import logger
from telegrams import create_bot_application

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')
warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')

def main():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(create_bot_application())
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Main function error: {e}")
    finally:
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
                
            if loop and not loop.is_closed():
                pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                if pending_tasks:
                    logger.info(f"Cancelling {len(pending_tasks)} pending tasks...")
                    for task in pending_tasks:
                        task.cancel()
                        
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
    
