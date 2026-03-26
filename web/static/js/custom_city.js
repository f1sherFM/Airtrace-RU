(function () {
    let notificationCounter = 0;

    function showNotification(message, type = "info") {
        const colors = {
            success: "bg-green-500/20 border-green-500/50 text-green-300",
            error: "bg-red-500/20 border-red-500/50 text-red-300",
            info: "bg-blue-500/20 border-blue-500/50 text-blue-300",
        };

        const liveRegion = document.getElementById("custom-notification-live");
        if (liveRegion) {
            liveRegion.textContent = message;
        }

        const notification = document.createElement("div");
        notificationCounter += 1;
        notification.className = `fixed top-4 right-4 ${colors[type]} border rounded-xl px-4 py-3 text-sm z-50 max-w-sm`;
        notification.setAttribute("role", "status");
        notification.setAttribute("aria-live", "polite");
        notification.setAttribute("id", `custom-notification-${notificationCounter}`);
        notification.innerHTML = `
            <div class="flex items-center gap-2">
                <i data-lucide="${type === "success" ? "check-circle" : type === "error" ? "x-circle" : "info"}" class="w-4 h-4"></i>
                <span>${message}</span>
            </div>
        `;

        document.body.appendChild(notification);
        if (typeof lucide !== "undefined") {
            lucide.createIcons();
        }
        setTimeout(() => notification.remove(), 3500);
    }

    function setDecimalRequired(isRequired) {
        const latInput = document.getElementById("lat");
        const lonInput = document.getElementById("lon");
        if (latInput) {
            latInput.required = isRequired;
        }
        if (lonInput) {
            lonInput.required = isRequired;
        }
    }

    function switchFormat(format) {
        const decimalInputs = document.getElementById("decimal-inputs");
        const dmsInputs = document.getElementById("dms-inputs");
        const textInputs = document.getElementById("text-inputs");
        const decimalBtn = document.getElementById("decimal-btn");
        const dmsBtn = document.getElementById("dms-btn");
        const textBtn = document.getElementById("text-btn");

        [decimalInputs, dmsInputs, textInputs].forEach((section) => section && section.classList.add("hidden"));
        [decimalBtn, dmsBtn, textBtn].forEach((button) => {
            if (!button) {
                return;
            }
            button.classList.remove("bg-blue-500/30", "border-blue-500/50");
            button.classList.add("bg-white/10", "border-white/20", "hover:bg-white/20");
        });

        if (format === "decimal") {
            decimalInputs?.classList.remove("hidden");
            decimalBtn?.classList.add("bg-blue-500/30", "border-blue-500/50");
            decimalBtn?.classList.remove("bg-white/10", "border-white/20", "hover:bg-white/20");
            setDecimalRequired(true);
        } else if (format === "dms") {
            dmsInputs?.classList.remove("hidden");
            dmsBtn?.classList.add("bg-blue-500/30", "border-blue-500/50");
            dmsBtn?.classList.remove("bg-white/10", "border-white/20", "hover:bg-white/20");
            setDecimalRequired(false);
        } else if (format === "text") {
            textInputs?.classList.remove("hidden");
            textBtn?.classList.add("bg-blue-500/30", "border-blue-500/50");
            textBtn?.classList.remove("bg-white/10", "border-white/20", "hover:bg-white/20");
            setDecimalRequired(false);
        }
    }

    function fillCoordinates(lat, lon, cityName) {
        document.getElementById("lat").value = lat;
        document.getElementById("lon").value = lon;
        document.getElementById("city_name").value = cityName;
        switchFormat("decimal");
    }

    function fillTextExample() {
        document.getElementById("coord-text").value = "61°15′00″ с. ш., 73°26′00″ в. д.";
        document.getElementById("city_name").value = "Сургут";
        switchFormat("text");
    }

    function fillDMSExample() {
        document.getElementById("lat-deg").value = 61;
        document.getElementById("lat-min").value = 15;
        document.getElementById("lat-sec").value = 0;
        document.getElementById("lat-dir").value = "N";
        document.getElementById("lon-deg").value = 73;
        document.getElementById("lon-min").value = 26;
        document.getElementById("lon-sec").value = 0;
        document.getElementById("lon-dir").value = "E";
        document.getElementById("city_name").value = "Сургут";
        switchFormat("dms");
    }

    function convertDMSToDecimal() {
        try {
            const latDeg = parseFloat(document.getElementById("lat-deg").value) || 0;
            const latMin = parseFloat(document.getElementById("lat-min").value) || 0;
            const latSec = parseFloat(document.getElementById("lat-sec").value) || 0;
            const latDir = document.getElementById("lat-dir").value;
            const lonDeg = parseFloat(document.getElementById("lon-deg").value) || 0;
            const lonMin = parseFloat(document.getElementById("lon-min").value) || 0;
            const lonSec = parseFloat(document.getElementById("lon-sec").value) || 0;
            const lonDir = document.getElementById("lon-dir").value;

            if (latDeg < 0 || latDeg > 90) {
                showNotification("Градусы широты должны быть от 0 до 90", "error");
                return;
            }
            if (lonDeg < 0 || lonDeg > 180) {
                showNotification("Градусы долготы должны быть от 0 до 180", "error");
                return;
            }
            if (latMin < 0 || latMin >= 60 || lonMin < 0 || lonMin >= 60) {
                showNotification("Минуты должны быть от 0 до 59", "error");
                return;
            }
            if (latSec < 0 || latSec >= 60 || lonSec < 0 || lonSec >= 60) {
                showNotification("Секунды должны быть от 0 до 59.999", "error");
                return;
            }

            let lat = latDeg + latMin / 60 + latSec / 3600;
            let lon = lonDeg + lonMin / 60 + lonSec / 3600;
            if (latDir === "S") {
                lat *= -1;
            }
            if (lonDir === "W") {
                lon *= -1;
            }

            lat = Math.round(lat * 10000) / 10000;
            lon = Math.round(lon * 10000) / 10000;
            document.getElementById("lat").value = lat;
            document.getElementById("lon").value = lon;
            switchFormat("decimal");
            showNotification(`Координаты конвертированы: ${lat}, ${lon}`, "success");
        } catch (error) {
            showNotification(`Ошибка при конвертации координат: ${error.message}`, "error");
        }
    }

    function parseTextCoordinates() {
        const text = document.getElementById("coord-text").value.trim();
        if (!text) {
            showNotification("Введите координаты для распознавания", "error");
            return;
        }

        try {
            let lat = null;
            let lon = null;
            const patterns = [
                /(\d+)°(\d+)[′'](\d+)[″"]?\s*([сcюns])[.\s]*ш[.\s]*,?\s*(\d+)°(\d+)[′'](\d+)[″"]?\s*([вveзw])[.\s]*д[.\s]*/i,
                /(\d+)°(\d+)'(\d+)"?\s*([ns]),?\s*(\d+)°(\d+)'(\d+)"?\s*([ew])/i,
                /(\d+)\s+(\d+)\s+(\d+)\s*([ns])\s+(\d+)\s+(\d+)\s+(\d+)\s*([ew])/i,
                /(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)/,
                /(\d+)°(\d+)[′']?\s*([сcюns])[.\s]*ш[.\s]*,?\s*(\d+)°(\d+)[′']?\s*([вveзw])[.\s]*д[.\s]*/i,
                /(\d+)°\s*(\d+)'?\s*([ns]),?\s*(\d+)°\s*(\d+)'?\s*([ew])/i,
            ];

            for (let index = 0; index < patterns.length; index += 1) {
                const match = text.match(patterns[index]);
                if (!match) {
                    continue;
                }

                if (index === 3) {
                    lat = parseFloat(match[1]);
                    lon = parseFloat(match[2]);
                } else if (index === 4 || index === 5) {
                    const latDeg = parseInt(match[1], 10);
                    const latMin = parseInt(match[2], 10);
                    const latDir = match[3].toLowerCase();
                    const lonDeg = parseInt(match[4], 10);
                    const lonMin = parseInt(match[5], 10);
                    const lonDir = match[6].toLowerCase();
                    lat = latDeg + latMin / 60;
                    lon = lonDeg + lonMin / 60;
                    if (latDir === "s" || latDir === "ю") {
                        lat *= -1;
                    }
                    if (lonDir === "w" || lonDir === "з") {
                        lon *= -1;
                    }
                } else {
                    const latDeg = parseInt(match[1], 10);
                    const latMin = parseInt(match[2], 10);
                    const latSec = parseInt(match[3], 10);
                    const latDir = match[4].toLowerCase();
                    const lonDeg = parseInt(match[5], 10);
                    const lonMin = parseInt(match[6], 10);
                    const lonSec = parseInt(match[7], 10);
                    const lonDir = match[8].toLowerCase();
                    lat = latDeg + latMin / 60 + latSec / 3600;
                    lon = lonDeg + lonMin / 60 + lonSec / 3600;
                    if (latDir === "s" || latDir === "ю") {
                        lat *= -1;
                    }
                    if (lonDir === "w" || lonDir === "з") {
                        lon *= -1;
                    }
                }
                break;
            }

            if (lat === null || lon === null) {
                showNotification("Не удалось распознать координаты. Попробуйте другой формат.", "error");
                return;
            }
            if (lat < -90 || lat > 90) {
                showNotification("Широта должна быть от -90 до 90", "error");
                return;
            }
            if (lon < -180 || lon > 180) {
                showNotification("Долгота должна быть от -180 до 180", "error");
                return;
            }

            lat = Math.round(lat * 10000) / 10000;
            lon = Math.round(lon * 10000) / 10000;
            document.getElementById("lat").value = lat;
            document.getElementById("lon").value = lon;
            switchFormat("decimal");
            showNotification(`Координаты распознаны: ${lat}, ${lon}`, "success");
        } catch (error) {
            showNotification(`Ошибка при распознавании координат: ${error.message}`, "error");
        }
    }

    function handleSubmit(event) {
        const form = event.currentTarget;
        const submitButton = form.querySelector('button[type="submit"]');
        const lat = parseFloat(document.getElementById("lat").value);
        const lon = parseFloat(document.getElementById("lon").value);

        if (Number.isNaN(lat) || Number.isNaN(lon)) {
            event.preventDefault();
            showNotification("Заполните координаты в десятичном формате перед отправкой формы.", "error");
            return;
        }
        if (lat < -90 || lat > 90) {
            event.preventDefault();
            showNotification("Широта должна быть в диапазоне от -90 до 90.", "error");
            document.getElementById("lat").focus();
            return;
        }
        if (lon < -180 || lon > 180) {
            event.preventDefault();
            showNotification("Долгота должна быть в диапазоне от -180 до 180.", "error");
            document.getElementById("lon").focus();
            return;
        }

        if (submitButton) {
            submitButton.disabled = true;
            submitButton.innerHTML = `
                <div class="flex items-center justify-center gap-2">
                    <div class="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span>Загрузка данных...</span>
                </div>
            `;
        }
    }

    function initializeCustomCityPage() {
        const form = document.getElementById("custom-city-form");
        if (!form) {
            return;
        }
        if (typeof lucide !== "undefined") {
            lucide.createIcons();
        }
        form.addEventListener("submit", handleSubmit);
        switchFormat("decimal");
    }

    window.fillCoordinates = fillCoordinates;
    window.fillTextExample = fillTextExample;
    window.fillDMSExample = fillDMSExample;
    window.switchFormat = switchFormat;
    window.convertDMSToDecimal = convertDMSToDecimal;
    window.parseTextCoordinates = parseTextCoordinates;

    document.addEventListener("DOMContentLoaded", initializeCustomCityPage);
})();
