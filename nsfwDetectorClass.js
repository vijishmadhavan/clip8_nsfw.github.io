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
        this._offScreenCanvas = document.createElement('canvas');
        this._offScreenCanvas.width = 124;
        this._offScreenCanvas.height = 124;
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
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        const img = await createImageBitmap(blob);
        const ctx = this._offScreenCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, this._offScreenCanvas.width, this._offScreenCanvas.height);
        return new Promise((resolve, reject) => {
            this._offScreenCanvas.toBlob(blob => {
                if (!blob) {
                    reject('Canvas to Blob conversion failed');
                    return;
                }
                const blobUrl = URL.createObjectURL(blob);
                resolve(blobUrl);
            }, 'image/jpeg');
        });
    }

    async isNsfwBulk(imageUrls) {
        this._imageQueue = imageUrls.map(url => fetch(url).then(response => response.blob()));
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
            semaphore[index] = imagePromise.then(blob => {
                return createImageBitmap(blob).then(img => {
                    const ctx = this._offScreenCanvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, this._offScreenCanvas.width, this._offScreenCanvas.height);
                    return new Promise((resolve, reject) => {
                        this._offScreenCanvas.toBlob(blob => {
                            if (!blob) {
                                reject('Canvas to Blob conversion failed');
                                return;
                            }
                            const blobUrl = URL.createObjectURL(blob);
                            resolve(blobUrl);
                        }, 'image/jpeg');
                    });
                });
            }).then(blobUrl => this.isNsfw(blobUrl)).then(result => {
                console.log(`Classification for ${blobUrl}:`, result ? 'NSFW' : 'Safe');
                if (!result) {
                    window.displayImage(blobUrl);
                }
                results.push({ imageUrl: blobUrl, isNsfw: result });
                return null;
            }).catch(error => {
                if (!error.message.includes('Unchecked runtime.lastError')) {
                    console.error(`Error processing image ${blobUrl}:`, error);
                    results.push({ imageUrl: blobUrl, error: error.toString() });
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
