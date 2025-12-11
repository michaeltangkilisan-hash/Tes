from flask import Flask, render_template, request
import threading

from EyeDect import run_detection

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    message = ""

    if request.method == "POST":
        kacamata = request.form.get("kacamata")  # "ya" atau "tidak"
        mata = request.form.get("mata")  # "1" atau "2"
        mode = request.form.get("mode", "1")  # "1" = RGB, "2" = grayscale manual

        # Jalankan deteksi mata di thread terpisah supaya request HTTP tidak hang
        threading.Thread(target=run_detection, args=(mode, mata), daemon=True).start()

        # Susun pesan informasi ke pengguna
        if mata == "1":
            mata_text = "Sistem dikonfigurasi untuk mendeteksi 1 mata (kiri atau kanan). "
        else:
            mata_text = "Sistem dikonfigurasi untuk mendeteksi 2 mata sekaligus. "

        kacamata_text = ""
        if kacamata == "ya":
            kacamata_text = (
                "Catatan: Anda memilih memakai kacamata. "
                "Hindari kacamata dengan lensa terlalu gelap karena dapat menurunkan akurasi deteksi. "
            )

        message = (
            "Sistem pendeteksi mata sedang dijalankan. "
            "Silakan lihat jendela kamera dan tekan tombol 'q' untuk keluar. "
            + mata_text
            + kacamata_text
        )

    return render_template("index.html", message=message)


if __name__ == "__main__":
    # Jalankan server Flask lokal
    app.run(debug=True)
