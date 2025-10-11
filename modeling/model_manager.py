from typing import Dict


class ModelManager:
    def __init__(self, trading_service: 'TradingService', model_path: str = 'models', redis_client: Optional[redis.Redis] = None):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.redis_client = redis_client
        self._cache: Dict[str, 'BaseModel'] = {}
        self._lock = asyncio.Lock()
        self._is_closed = False
        self.logger = logger
        self._training_executor = None
        self._prediction_executor = None
        self.trading_service = trading_service

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError("ModelManager has been closed and cannot be used")

    async def _get_executor(self, executor_type: str) -> ThreadPoolExecutor:
        self._check_if_closed()
        
        if executor_type == 'training':
            if self._training_executor is None or self._training_executor._shutdown:
                max_workers = max(1, (os.cpu_count() or 4) // 2)
                self._training_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='ModelTrainingExecutor')
            return self._training_executor
        elif executor_type == 'prediction':
            if self._prediction_executor is None or self._prediction_executor._shutdown:
                max_workers = max(1, (os.cpu_count() or 2))
                self._prediction_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='ModelPredictionExecutor')
            return self._prediction_executor
        else:
            raise ValueError("Invalid executor type specified.")

    async def _run_in_executor(self, func, *args, executor_type: str = 'prediction', **kwargs) -> Any:
        self._check_if_closed()
        loop = asyncio.get_running_loop()
        executor = await self._get_executor(executor_type)
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    def _get_model_class(self, model_type: str) -> Type['BaseModel']:
        if model_type == 'lstm':
            return LSTMModel
        if model_type == 'xgboost':
            return XGBoostModel
        raise ValueError(f"Unknown model type: {model_type}")

    async def get_model(self, model_type: str, symbol: str, timeframe: str, auto_load: bool = True) -> Optional['BaseModel']:
        self._check_if_closed()
        key = f"{model_type}-{symbol}-{timeframe}"
        
        async with self._lock:
            if key in self._cache:
                cached_model = self._cache[key]
                if not cached_model._is_closed:
                    return cached_model
                del self._cache[key]
        
        model_dir = self.model_path / model_type
        model_class = self._get_model_class(model_type)
        model_path_str = str(model_dir)

        try:
            model = await self._run_in_executor(lambda: model_class(symbol=symbol, timeframe=timeframe, model_path=model_path_str), executor_type='prediction')
            
            if auto_load:
                if model_type == 'lstm':
                    model_file, scaler_file = model._get_model_paths(model_type, "keras")
                else:
                    model_file, scaler_file = model._get_model_paths(model_type, "json")
                
                if model_file.exists() and (scaler_file.exists() or model_type == 'xgboost'):
                    try:
                        await self._run_in_executor(model.load, executor_type='prediction')
                        self.logger.info(f"Loaded existing {model_type.upper()} model for {key}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_type.upper()} model for {key}: {e}")
                        model.is_trained = False
                else:
                    self.logger.info(f"No existing {model_type.upper()} model found for {key}, needs training")
                    model.is_trained = False

            async with self._lock:
                if key not in self._cache:
                    self._cache[key] = model
                else:
                    model.cleanup()
                    return self._cache[key]
            return model
        except Exception as e:
            self.logger.error(f"Could not get or create {model_type.upper()} model for {key}: {e}", exc_info=True)
            return None

    async def train_model(self, model_type: str, symbol: str, timeframe: str, data: pd.DataFrame, **kwargs) -> bool:
        self._check_if_closed()
        
        if data is None or data.empty:
            self.logger.warning(f"Cannot train {model_type} for {symbol}-{timeframe}: empty data")
            return False
        
        model = await self.get_model(model_type, symbol, timeframe, auto_load=False)
        if not model:
            self.logger.error(f"Failed to create model {model_type} for {symbol}-{timeframe}")
            return False
        
        try:
            success = await self._run_in_executor(model.fit, data, executor_type='training', **kwargs)
            if success:
                self.logger.info(f"Successfully trained {model_type.upper()} model for {symbol}-{timeframe}")
                async with self._lock:
                    key = f"{model_type}-{symbol}-{timeframe}"
                    self._cache[key] = model
                return True
            else:
                self.logger.warning(f"Training failed for {model_type.upper()} model {symbol}-{timeframe}")
                return False
        except Exception as e:
            self.logger.error(f"Error training {model_type.upper()} model for {symbol}-{timeframe}: {e}", exc_info=True)
            return False

    async def initialize_all_models(self):
        self._check_if_closed()
        self.logger.info("Initializing all models (LSTM, XGBoost)...")
        tasks = []
        model_files = []
        
        for model_type_dir in self.model_path.iterdir():
            if not model_type_dir.is_dir():
                continue
            
            model_type = model_type_dir.name
            if model_type == "lstm": 
                ext = "keras"
            elif model_type == "xgboost": 
                ext = "json"
            else: 
                continue
            
            for file in model_type_dir.glob(f"*.{ext}"):
                model_files.append((file, model_type))

        for file, model_type in model_files:
            try:
                base_name = file.stem
                if model_type == 'lstm': 
                    parts_str = base_name.replace("lstm_", "")
                elif model_type == 'xgboost': 
                    parts_str = base_name.replace("xgboost_", "")
                else: 
                    continue
                
                symbol, timeframe = parts_str.rsplit('_', 1)
                symbol_formatted = symbol.upper()
                if '/' not in symbol_formatted and len(symbol_formatted) > 4:
                    symbol_formatted = symbol_formatted.replace("USDT", "/USDT", 1)

                tasks.append(self.get_model(model_type, symbol_formatted, timeframe))
            except (IndexError, ValueError) as e:
                self.logger.warning(f"Could not parse model file name: {file.name}. Error: {e}")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Error during model initialization: {res}", exc_info=res)
        self.logger.info(f"Model initialization finished. Loaded/Created {len(tasks)} models.")

    async def train_and_save_model(self, model_type: str, symbol: str, timeframe: str) -> bool:
        self.logger.info(f"Attempting to train and save {model_type} model for {symbol}-{timeframe}")
        data = await self.trading_service.get_data_for_model(symbol, timeframe)
        if data is None or data.empty:
            self.logger.warning(f"Could not fetch sufficient data for model training: {symbol}-{timeframe}")
            return False
        
        return await self.train_model(model_type, symbol, timeframe, data)

    async def predict_with_confidence(self, model_type: str, symbol: str, timeframe: str) -> Optional[Dict[str, float]]:
        self._check_if_closed()
        
        model = await self.get_model(model_type, symbol, timeframe)
        if not model:
            self.logger.error(f"Failed to get model for prediction: {model_type} on {symbol}-{timeframe}")
            return None

        if not model.is_trained:
            self.logger.info(f"Model {model_type} for {symbol}-{timeframe} is not trained. Attempting to train now.")
            success = await self.train_and_save_model(model_type, symbol, timeframe)
            if not success:
                self.logger.warning(f"Training failed for {model_type} on {symbol}-{timeframe}. Skipping prediction.")
                return None
            model = await self.get_model(model_type, symbol, timeframe)
            if not model or not model.is_trained:
                self.logger.error(f"Model still not trained after attempt for {model_type} on {symbol}-{timeframe}.")
                return None

        data = await self.trading_service.get_data_for_model(symbol, timeframe, for_prediction=True)
        if data is None or data.empty:
            self.logger.warning(f"Empty data for prediction {model_type} on {symbol}-{timeframe}")
            return None

        try:
            prediction_result = await self._run_in_executor(model.predict, data, executor_type='prediction')
            if prediction_result is None or len(prediction_result) != 2:
                return None
            
            prediction, uncertainty = prediction_result
            if prediction is None or len(prediction) == 0 or np.isnan(prediction[0]) or np.isinf(prediction[0]):
                return None
            
            raw_confidence = 50.0
            calibrated_confidence = 50.0
            
            current_price = data['close'].iloc[-1]
            if current_price > 0:
                price_change_pct = abs(prediction[0] - current_price) / current_price * 100
                raw_confidence = min(price_change_pct * 10, 100.0)
                calibrated_confidence = raw_confidence
            
            final_confidence = max(0.0, min(100.0, calibrated_confidence * (1 - uncertainty ** 2)))
            
            return {
                'prediction': float(prediction[0]),
                'confidence': final_confidence,
                'raw_confidence': raw_confidence,
                'uncertainty': uncertainty
            }
        except (RuntimeError, CancelledError) as e:
            self.logger.warning(f"Prediction task cancelled or failed for {model_type} on {symbol}-{timeframe}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_type} on {symbol}-{timeframe}: {e}")
            return None

    async def record_signal_performance(self, model_type: str, symbol: str, timeframe: str, success: bool):
        self._check_if_closed()
        model = await self.get_model(model_type, symbol, timeframe)
        if model and hasattr(model, 'calibrator') and model.calibrator:
            confidence = 0.5
            model.calibrator.add_prediction(f"{model_type}_{symbol}_{timeframe}", confidence, success)

    async def shutdown(self):
        if self._is_closed:
            return
        
        self._is_closed = True
        self.logger.info("Shutting down ModelManager...")
        
        async with self._lock:
            models_to_clean = list(self._cache.values())
            self._cache.clear()
        
        for model in models_to_clean:
            try:
                model.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up model {model.symbol}-{model.timeframe}: {e}")
        
        for executor_attr in ['_training_executor', '_prediction_executor']:
            if hasattr(self, executor_attr):
                executor = getattr(self, executor_attr)
                if executor:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except Exception as e:
                        self.logger.error(f"Error shutting down executor: {e}")
        
        self.logger.info("ModelManager shut down.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
        return False