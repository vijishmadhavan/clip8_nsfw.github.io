class NsfwDetector {
    constructor() {
        this._threshold = 0.30;
        this._nsfwLabels = [
            'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED',
            'MALE_GENITALIA_EXPOSED', 'BLOOD_SHED', 'VIOLENCE', 'GORE', 'PORNOGRAPHY', 'DRUGS', 'ALCOHOL',
            'CHILD_PORN', 'CHILD_KISS', 'CHILD_VULGARITY', 'INAPROPRIATE_CLOTHING', 'SENSUAL_KISS',
        ];
        this._classifierPromise = window.tensorflowPipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
        this._cache = {};
        this._imageQueue = [];
        this._modelReady = false;
        this._concurrencyLimit = navigator.hardwareConcurrency || 5;
        this._classifierPromise.then(() => {
            this._modelReady = true;
            console.log('Model is downloaded and ready to use.');
            this._processQueuedImages();
        }).catch(error => {
            console.error('Error loading the model:', error);
        });
    }

    async isNsfw(imageUrl) {
        if (this._cache[imageUrl] !== undefined) {
            return this._cache[imageUrl];
        }

        let blobUrl = '';
        try {
            blobUrl = await this._loadAndResizeImage(imageUrl);
            const classifier = await this._classifierPromise;
            const output = await classifier(blobUrl, this._nsfwLabels);
            const nsfwDetected = output.some(result => result.score > this._threshold);
            console.log(`Classification for ${imageUrl}:`, nsfwDetected ? 'NSFW' : 'Safe');
            console.log('Detailed classification results:', output);
            this._cache[imageUrl] = nsfwDetected;
            return nsfwDetected;
        } catch (error) {
            if (!error.message.includes('Unchecked runtime.lastError')) {
                console.error('Error during NSFW classification: ', error);
            }
            return false;
        } finally {
            if (blobUrl) {
                URL.revokeObjectURL(blobUrl);
            }
        }
    }

    async _loadAndResizeImage(imageUrl) {
        const img = await this._loadImage(imageUrl);
        const offScreenCanvas = document.createElement('canvas');
        const ctx = offScreenCanvas.getContext('2d');
        offScreenCanvas.width = 124;
        offScreenCanvas.height = 124;
        ctx.drawImage(img, 0, 0, offScreenCanvas.width, offScreenCanvas.height);
        return new Promise((resolve, reject) => {
            offScreenCanvas.toBlob(blob => {
                if (!blob) {
                    reject('Canvas to Blob conversion failed');
                    return;
                }
                const blobUrl = URL.createObjectURL(blob);
                resolve(blobUrl);
            }, 'image/jpeg');
        });
    }

    async _loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                img.decode().then(() => resolve(img)).catch(error => {
                    if (!error.message.includes('Unchecked runtime.lastError')) {
                        reject(error);
                    } else {
                        resolve(img);
                    }
                });
            };
            img.onerror = () => reject(`Failed to load image: ${url}`);
            img.src = url;
        });
    }

    async isNsfwBulk(imageUrls) {
        this._imageQueue = imageUrls.map(url => this._loadImage(url));
        if (!this._modelReady) {
            console.log('Model is not ready yet. Queuing images for classification.');
            return;
        }
        return this._processQueuedImages();
    }

    async _processQueuedImages() {
        const semaphore = new Array(this._concurrencyLimit).fill(Promise.resolve());
        const results = [];
        const processImage = async (imagePromise) => {
            const index = await Promise.race(semaphore.map((p, index) => p.then(() => index)));
            semaphore[index] = imagePromise.then(img => this._loadAndResizeImage(img.src)).then(blobUrl => this.isNsfw(blobUrl)).then(result => {
                console.log(`Classification for ${img.src}:`, result ? 'NSFW' : 'Safe');
                if (!result) {
                    window.displayImage(img.src);
                }
                results.push({ imageUrl: img.src, isNsfw: result });
                return null;
            }).catch(error => {
                if (!error.message.includes('Unchecked runtime.lastError')) {
                    console.error(`Error processing image ${img.src}:`, error);
                    results.push({ imageUrl: img.src, error: error.toString() });
                }
                return null;
            });
        };

        await Promise.all(this._imageQueue.map(processImage));
        await Promise.all(semaphore);
        this._imageQueue = [];
        return results;
    }
}

window.NsfwDetector = NsfwDetector;
