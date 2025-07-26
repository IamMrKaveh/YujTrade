from imports import SYMBOLS, logger, asyncio

async def _process_all_symbols_in_batches(analysis_stats):
    """پردازش تمام نمادها به صورت دسته‌ای"""
    all_signals = []
    batch_size = min(3, len(SYMBOLS))
    total_batches = (len(SYMBOLS) - 1) // batch_size + 1
    
    logger.info(f"Processing {len(SYMBOLS)} symbols in {total_batches} batches of size {batch_size}")
    
    for i in range(0, len(SYMBOLS), batch_size):
        try:
            batch_symbols = SYMBOLS[i:i+batch_size]
            current_batch = i // batch_size + 1
            
            batch_signals = await _process_single_batch(
                batch_symbols, current_batch, total_batches, analysis_stats
            )
            all_signals.extend(batch_signals)
            
            # تاخیر بین دسته‌ها
            if i + batch_size < len(SYMBOLS):
                await asyncio.sleep(3)
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            analysis_stats['failed'] += len(batch_symbols) if 'batch_symbols' in locals() else batch_size
    
    return all_signals

async def _process_single_batch(batch_symbols, current_batch, total_batches, analysis_stats):
    """پردازش یک دسته از نمادها"""
    logger.info(f"Processing batch {current_batch}/{total_batches}")
    
    batch_tasks = _create_batch_tasks(batch_symbols)
    if not batch_tasks:
        logger.warning(f"No valid symbols in batch {current_batch}")
        analysis_stats['failed'] += len(batch_symbols)
        logger.debug(f"Batch {current_batch} processing failed - no valid tasks created")
        return []
    
    result = await _execute_batch_analysis(batch_tasks, batch_symbols, current_batch, analysis_stats)
    return result

def _create_batch_tasks(batch_symbols):
    """ایجاد تسک‌های تحلیل برای دسته"""
    batch_tasks = []
    for symbol in batch_symbols:
        if symbol:  # Ensure symbol is not None or empty
            batch_tasks.append(_analyze_single_symbol(symbol))
    
    logger.debug(f"Created {len(batch_tasks)} analysis tasks from {len(batch_symbols)} symbols")
    return batch_tasks

async def _execute_batch_analysis(batch_tasks, batch_symbols, current_batch, analysis_stats):
    """اجرای تحلیل دسته‌ای با مدیریت خطا"""
    try:
        batch_results = await asyncio.wait_for(
            asyncio.gather(*batch_tasks, return_exceptions=True),
            timeout=300  # 5 دقیقه برای هر دسته
        )
        
        result = _process_batch_results(batch_results, batch_symbols, analysis_stats)
        logger.debug(f"Batch {current_batch} analysis executed successfully, got {len(result)} valid signals")
        return result
        
    except asyncio.TimeoutError:
        logger.warning(f"Batch {current_batch} timed out")
        analysis_stats['failed'] += len(batch_symbols)
        logger.debug(f"Batch {current_batch} execution failed due to timeout")
        return []
    except Exception as e:
        logger.error(f"Error processing batch {current_batch}: {e}")
        analysis_stats['failed'] += len(batch_symbols)
        return []

def _process_batch_results(batch_results, batch_symbols, analysis_stats):
    """پردازش نتایج دسته‌ای"""
    signals = []
    
    for j, result in enumerate(batch_results):
        if isinstance(result, Exception):
            analysis_stats['failed'] += 1
            symbol_name = batch_symbols[j] if j < len(batch_symbols) else "Unknown"
            logger.error(f"Analysis error for {symbol_name}: {result}")
        elif result is not None and isinstance(result, dict):
            signals.append(result)
            analysis_stats['successful'] += 1
        else:
            analysis_stats['failed'] += 1
    
    logger.debug(f"Batch results processed: {len(signals)} valid signals from {len(batch_results)} results")
    return signals

