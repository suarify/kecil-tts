"""Build the Phase 4 text pool (plan section 8.1).

Outputs (data/text/):
  corpus_train.tsv   id, category, text, norm, teacher, phonemes   (~7,500 prompts, ~9 h raw)
  test_200.tsv       held out, never trained on
  val_100.tsv        held out validation
  OOD_texts.txt      3,000 unused phonemised sentences (the trainer needs this file)

Mix: 35% general (Tatoeba + Wikipedia), 30% conversational/call-centre (templates),
15% numbers, 10% code-switch, 5% questions/exclamations, 5% short replies (paired).
Selection inside each category is greedy for phoneme-pair coverage.
"""
import bz2
import collections
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from build_teacher_respell import respell  # noqa: E402
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

RNG = random.Random(2026)
G = G2P(stress=False, r='r', diph='glide')
RESPELL = json.load(open(os.path.join(ROOT, 'g2p', 'teacher_respell.json'), encoding='utf-8'))
TARGET = 7500
SHARE = {'general': .35, 'conversational': .30, 'numbers': .15, 'codeswitch': .10,
         'question': .05, 'short': .05}
OK_CHARS = re.compile(r"^[A-Za-z0-9 ,.?!'%/:-]+$")

NAMES_F = ['Siti', 'Aminah', 'Nurul', 'Aisyah', 'Farah', 'Lina', 'Zarina', 'Hani', 'Mei Ling', 'Priya',
           'Nadia', 'Liyana', 'Syafiqah', 'Kavitha', 'Hui Min', 'Rosnah', 'Fatimah', 'Intan', 'Sarah', 'Alya']
NAMES_M = ['Ahmad', 'Farid', 'Hafiz', 'Zul', 'Kamal', 'Rizal', 'Kumar', 'Wei Jie', 'Azman', 'Hakim',
           'Danial', 'Imran', 'Ravi', 'Jason', 'Faizal', 'Amir', 'Syahmi', 'Arif', 'Kok Leong', 'Ismail']
FOREIGN = {'Tom': 'Ahmad', 'Mary': 'Siti', 'John': 'Hafiz', 'Ken': 'Kamal', 'Jim': 'Rizal', 'Mike': 'Farid',
           'Jack': 'Zul', 'Bob': 'Azman', 'Alice': 'Aminah', 'Jane': 'Nurul', 'Sami': 'Imran', 'Layla': 'Liyana',
           'Muiriel': 'Farah', 'Tatoeba': 'laman ini'}

# ---------------------------------------------------------------- general
def tatoeba():
    test = {l.split('\t')[2].strip() for l in open(os.path.join(ROOT, 'eval', 'test_40.tsv'), encoding='utf-8').read().splitlines()[1:]}
    out = []
    for line in bz2.open(os.path.join(ROOT, 'data', 'text', 'src', 'zsm_sentences.tsv.bz2'), 'rt', encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) < 3:
            continue
        s = p[2].strip()
        for a, b in FOREIGN.items():
            s = re.sub(rf'\b{a}\b', b, s)
        if s in test or not OK_CHARS.match(s) or re.search(r'\b(Mr|Mrs|Ms)\b', s):
            continue
        if 3 <= len(s.split()) <= 22:
            out.append(s)
    return out


def wikipedia(limit=40000):
    import pandas as pd
    p = os.path.join(ROOT, 'data', 'text', 'src', 'wiki', '20231101.ms', 'train-00000-of-00001.parquet')
    d = pd.read_parquet(p, columns=['text']).sample(frac=1, random_state=7)
    shapes, out = set(), []
    for text in d.text:
        for para in text.split('\n'):
            if len(para) < 60 or any(c in para for c in '|{}[]=<>()"*#;'):
                continue
            for s in re.split(r'(?<=[.!?])\s+', para):
                s = s.strip()
                w = s.split()
                if not (6 <= len(w) <= 20) or not s.endswith('.') or not OK_CHARS.match(s):
                    continue
                caps = sum(1 for x in w[1:] if x[0].isupper())
                if caps / len(w) > 0.2 or re.search(r'\d{3,}', s):
                    continue
                shape = ' '.join('X' if x[0].isupper() or x[0].isdigit() else x for x in w)[:60]
                if shape in shapes:
                    continue
                shapes.add(shape)
                out.append(s)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- conversational
CONV = '''Selamat pagi {g} {n}, saya {a} dari {org}.
Selamat petang, terima kasih kerana menghubungi {org}.
Assalamualaikum {g} {n}, boleh saya bercakap sebentar?
Saya ingin mengingatkan {g} tentang temu janji pada hari {h} ini.
Temu janji {g} dijadualkan pada hari {h}, pukul {t}.
Adakah {g} masih boleh hadir pada masa tersebut?
Baik, saya akan kemas kini rekod {g} sekarang.
Mohon maaf atas kesulitan yang {g} alami.
Kami akan menghubungi {g} semula dalam masa dua puluh empat jam.
Boleh {g} sahkan nama penuh dan nombor kad pengenalan?
Terima kasih, maklumat {g} telah kami terima.
Sila tunggu sebentar, saya sedang menyemak sistem.
Maaf kerana membuat {g} menunggu.
Permohonan {g} sedang diproses dan akan siap dalam tiga hari bekerja.
Bayaran bulanan {g} sudah melepasi tarikh akhir.
Adakah {g} mahu saya hantar peringatan melalui mesej?
Kami ada tawaran istimewa untuk pelanggan setia seperti {g}.
Bolehkah saya tahu masa yang sesuai untuk menghubungi {g} semula?
Baiklah, saya akan telefon {g} semula pada hari {h}.
Saya faham kebimbangan {g}, biar saya bantu.
Pesanan {g} akan dihantar ke {tp} esok pagi.
Penghantaran mungkin lewat sedikit kerana cuaca buruk.
Kalau ada sebarang pertanyaan, {g} boleh hubungi kami pada bila-bila masa.
Terima kasih atas masa {g}, semoga hari {g} menyenangkan.
Klinik kami dibuka dari hari Isnin hingga Sabtu.
Doktor {n} akan berjumpa {g} di bilik nombor tiga.
Sila datang lima belas minit lebih awal untuk pendaftaran.
Jangan lupa bawa kad pengenalan dan surat rujukan.
Polisi insurans {g} akan tamat tempoh pada hujung bulan ini.
Adakah {g} berminat untuk memperbaharui polisi tersebut?
Saya boleh terangkan pilihan pelan yang ada sekarang.
Kami telah menerima aduan {g} dan sedang menyiasatnya.
Juruteknik kami akan sampai di {tp} antara pukul {t} hingga tengah hari.
Bekalan elektrik di kawasan {g} akan dipulihkan tidak lama lagi.
Sila pastikan ada orang dewasa di rumah semasa lawatan.
Kelas {g} bermula minggu depan di {tp}.
Yuran pendaftaran boleh dibayar di kaunter atau dalam talian.
Selamat datang ke {org}, bagaimana saya boleh membantu?
Boleh {g} ulang sekali lagi? Talian kurang jelas.
Suara {g} terputus-putus, saya akan cuba hubungi semula.
Saya pindahkan panggilan {g} kepada bahagian yang berkenaan.
Untuk keselamatan, panggilan ini mungkin dirakam.
Tempahan meja untuk {k} orang pada malam {h} telah disahkan.
Kami simpan meja {g} sehingga lima belas minit selepas waktu tempahan.
Bilik {g} sudah siap, {g} boleh daftar masuk sekarang.
Sarapan disediakan dari pukul tujuh hingga sepuluh pagi.
Kereta {g} sudah siap diservis dan boleh diambil petang ini.
Kami jumpa sedikit masalah pada brek, perlu ditukar.
Kos pembaikan dianggarkan dalam lingkungan tiga ratus ringgit.
Adakah {g} setuju untuk kami teruskan kerja pembaikan?
{a} di sini, saya mewakili pasukan khidmat pelanggan.
Mohon {g} jangan berkongsi kata laluan dengan sesiapa.
Kami tidak akan sesekali meminta nombor pin {g} melalui telefon.
Akaun {g} telah dikunci sementara untuk keselamatan.
Sila ke cawangan terdekat bersama dokumen pengenalan diri.
Anak {g} telah berjaya didaftarkan untuk sesi akan datang.
Mesyuarat ditunda ke hari {h} atas sebab yang tidak dapat dielakkan.
Terima kasih kerana bersabar, masalah {g} sudah selesai.
Ada apa-apa lagi yang boleh saya bantu hari ini?
Baik {g}, saya catatkan permintaan itu.
Sebentar ya {g}, saya semak dahulu.
Itu sahaja daripada saya, terima kasih banyak-banyak.
Kami hargai maklum balas daripada {g}.
Maaf, nombor yang {g} dail tidak dapat dihubungi.
Sila tinggalkan pesanan selepas bunyi isyarat.
Pejabat kami ditutup pada hari Ahad dan cuti umum.
Waktu operasi kami bermula pukul {t}.
{g} boleh buat tempahan melalui laman sesawang kami.
Kami akan hantar pengesahan sebaik sahaja tempahan siap.
Ubat {g} sudah siap, boleh diambil di farmasi.
Makan ubat ini dua kali sehari selepas makan.
Jangan lupa minum air secukupnya dan berehat.
Keputusan ujian darah {g} akan keluar dalam seminggu.
Saya {a}, pegawai yang menguruskan kes {g}.
Sila lengkapkan borang ini dan serahkan semula kepada kami.
Pembayaran balik akan dikreditkan ke akaun {g} dalam tujuh hari.
Hadiah {g} boleh dituntut di kaunter utama.
Tahniah {g}, permohonan anda telah diluluskan.
Harap maaf, permohonan {g} tidak berjaya kali ini.
{g} boleh memohon semula selepas tiga bulan.
Pemandu kami sedang dalam perjalanan ke {tp}.
Pemandu akan tiba dalam masa lebih kurang sepuluh minit.
Sila tunggu di pintu masuk utama.
Barang {g} telah tiba di pusat pengedaran.
Saya ucapkan selamat hari raya kepada {g} sekeluarga.
Saya hubungi {g} berkenaan bil {svc} yang masih tertunggak.
Boleh {g} jelaskan bayaran sebelum hari {h}?
{g} boleh membuat bayaran di mana-mana cawangan {org}.
Status permohonan {svc} {g} kini sedang disemak.
Pegawai kami akan melawat {tp} pada hari {h}.
Kami ingin menjemput {g} ke majlis pelancaran di {tp}.
Majlis akan bermula tepat pukul {t}, jangan lewat ya.
Sila maklumkan kehadiran {g} sebelum hari {h}.
Tempat letak kereta disediakan di bangunan sebelah.
Maaf {g}, slot pukul {t} sudah penuh.
Slot yang masih kosong ialah hari {h}, pukul {t}.
Saya cadangkan {g} datang pada waktu pagi kerana kurang sesak.
Kad {svc} baharu {g} akan dihantar melalui pos.
Sila aktifkan kad dalam masa tiga puluh hari.
Kami dapati ada transaksi luar biasa dalam akaun {g}.
Adakah {g} membuat transaksi di {tp} semalam?
Untuk keselamatan, kami telah menyekat kad tersebut.
Kad gantian akan siap dalam masa lima hari bekerja.
Pakej {svc} {g} akan dinaik taraf secara percuma.
Kelajuan baharu akan berkuat kuasa mulai hari {h}.
{g} tidak perlu membuat apa-apa, semuanya automatik.
Gangguan perkhidmatan di {tp} sedang dibaiki oleh pasukan kami.
Kami jangka perkhidmatan pulih sebelum pukul {t}.
Terima kasih kerana memilih {org}.
Kami gembira dapat berkhidmat untuk {g}.
Hari ini hari terakhir untuk menebus mata ganjaran {g}.
Mata ganjaran boleh ditukar dengan baucar belian.
Anak {g} tidak hadir ke sekolah hari ini, adakah dia sihat?
Mesyuarat ibu bapa dan guru akan diadakan pada hari {h}.
Keputusan peperiksaan boleh disemak mulai hari {h}.
Sila bawa anak {g} untuk suntikan susulan.
Rawatan susulan {g} dijadualkan di {org}.
Saya {a}, jururawat yang bertugas hari ini.
Doktor minta {g} berpuasa lapan jam sebelum ujian.
Jangan makan atau minum selepas tengah malam.
Ubat ini mungkin menyebabkan rasa mengantuk.
Sila elakkan memandu selepas mengambil ubat.
Kalau demam berlarutan, sila datang semula ke klinik.
Pesanan makanan {g} sedang disediakan di dapur.
Makanan akan sampai dalam masa tiga puluh minit.
Penghantar kami tidak dapat mencari alamat {g}.
Boleh {g} berikan tanda tempat yang berdekatan?
Bungkusan {g} telah ditinggalkan di pengawal keselamatan.
Sila tandatangan sebagai bukti penerimaan barang.
Saya dari jabatan sumber manusia {org}.
Temu duga {g} dijadualkan pada hari {h}, pukul {t}.
Sila bawa salinan sijil dan resume terkini.
Tahniah, {g} berjaya ke peringkat seterusnya.
Kami akan maklumkan keputusan dalam masa dua minggu.
Gaji akan dikreditkan pada setiap hujung bulan.
Cuti tahunan {g} masih berbaki lapan hari.
Permohonan cuti {g} telah diluluskan oleh penyelia.
Rumah yang {g} minati masih belum dijual.
Pemilik bersetuju untuk berbincang tentang harga.
Bila {g} senang untuk melihat rumah tersebut?
Ejen kami akan menunggu {g} di {tp}.
Kontrak sewa perlu ditandatangani sebelum hari {h}.
Deposit dua bulan perlu dibayar terlebih dahulu.
Sila pastikan kenderaan {g} dikunci dengan selamat.
Tayar kereta {g} perlu ditukar tidak lama lagi.
Cukai jalan {g} akan tamat bulan depan.
Pemeriksaan kenderaan boleh dibuat di {tp}.
Penerbangan {g} ke {tp} ditunda selama satu jam.
Sila daftar masuk di kaunter nombor lima.
Bagasi {g} tidak boleh melebihi had berat yang ditetapkan.
Pintu berlepas akan ditutup dua puluh minit sebelum penerbangan.
Selamat pulang ke tanah air, {g}.
Kami mohon {g} memberi penilaian terhadap perkhidmatan hari ini.
Pada skala satu hingga lima, berapa markah yang {g} berikan?
Terima kasih atas penilaian yang diberikan.'''.splitlines()
ORGS = ['Klinik Sejahtera', 'Bank Rakyat', 'Pusat Servis Maju', 'Hotel Seri Pantai', 'Syarikat Insurans Amanah',
        'Kedai Elektrik Jaya', 'Hospital Pakar Damai', 'Pejabat Tanah', 'Restoran Selera Kampung', 'Kolej Bestari']
PLACES = ['Shah Alam', 'Petaling Jaya', 'Johor Bahru', 'Kota Bharu', 'Ipoh', 'Melaka', 'Kuching', 'Kota Kinabalu',
          'Seremban', 'Kuantan', 'Alor Setar', 'Bangi', 'Cyberjaya', 'Putrajaya', 'Klang', 'Kajang']
DAYS = ['Isnin', 'Selasa', 'Rabu', 'Khamis', 'Jumaat', 'Sabtu', 'Ahad']
TIMES = ['sembilan pagi', 'sepuluh pagi', 'sebelas setengah pagi', 'dua petang', 'tiga petang', 'empat setengah petang',
         'lapan malam', 'lapan pagi']
TITLES = ['encik', 'puan', 'cik', 'tuan', 'Dato', 'doktor']


def fill(t):
    female = RNG.random() < 0.5
    n = RNG.choice(NAMES_F if female else NAMES_M)
    g = RNG.choice(['puan', 'cik'] if female else ['encik', 'tuan']) if RNG.random() < .8 else RNG.choice(TITLES)
    s = t.format(g=g, n=n, a=RNG.choice(NAMES_F + NAMES_M), org=RNG.choice(ORGS), h=RNG.choice(DAYS),
                 t=RNG.choice(TIMES), tp=RNG.choice(PLACES), k=RNG.choice(['dua', 'tiga', 'empat', 'enam']),
                 svc=RNG.choice(['internet', 'elektrik', 'air', 'telefon', 'kad kredit', 'pinjaman', 'insurans', 'astro']))
    return s[0].upper() + s[1:]


def conversational(n):
    return list({fill(RNG.choice(CONV)) for _ in range(n * 6)})


# ---------------------------------------------------------------- numbers
def rm():
    r = RNG.random()
    if r < .4:
        return f'RM{RNG.randint(1, 999)}'
    if r < .7:
        return f'RM{RNG.randint(1, 9999)}.{RNG.randint(0, 99):02d}'
    if r < .9:
        return f'RM{RNG.randint(1, 99)},{RNG.randint(0, 999):03d}'
    return f'RM{RNG.randint(1, 50)} juta'


def date():
    d, m, y = RNG.randint(1, 28), RNG.randint(1, 12), RNG.randint(1990, 2030)
    MON = ['Januari', 'Februari', 'Mac', 'April', 'Mei', 'Jun', 'Julai', 'Ogos', 'September', 'Oktober', 'November', 'Disember']
    return RNG.choice([f'{d}/{m}/{y}', f'{d} {MON[m - 1]} {y}', f'{d} {MON[m - 1]}'])


def time_():
    h, m = RNG.randint(1, 12), RNG.choice([0, 15, 30, 45, RNG.randint(1, 59)])
    return RNG.choice([f'{h}:{m:02d} pagi', f'{h}:{m:02d} petang', f'{h}.{m:02d} malam', f'{h}:{m:02d}', f'{h}pm', f'{h}am'])


def phone():
    return RNG.choice([f'01{RNG.randint(0, 9)}-{RNG.randint(100, 999)} {RNG.randint(1000, 9999)}',
                       f'03-{RNG.randint(1000, 9999)} {RNG.randint(1000, 9999)}'])


NUMT = ['Jumlah bayaran anda ialah {rm}.', 'Baki akaun anda sekarang {rm}.', 'Harga barang ini {rm} sahaja.',
        'Anda perlu membayar {rm} sebelum {date}.', 'Bil elektrik bulan ini berjumlah {rm}.',
        'Temu janji anda pada {date}, pukul {time}.', 'Kami akan sampai pada pukul {time}.',
        'Sila hubungi kami di talian {phone}.', 'Nombor telefon pejabat ialah {phone}.',
        'Kod pengesahan anda ialah {otp}.', 'Nombor tempahan anda {otp}.',
        'Diskaun sebanyak {pct}% untuk semua barang.', 'Kadar faedah tahunan ialah {dec}%.',
        'Suhu hari ini mencecah {temp} darjah Celsius.', 'Jarak ke sana kira-kira {km} km.',
        'Berat bungkusan itu {kg} kg.', 'Dia lahir pada {date}.', 'Kedai dibuka dari {h1} hingga {h2}.',
        'Sebanyak {big} orang hadir ke majlis itu.', 'Ini kali ke-{ord} saya datang ke sini.',
        'Penerbangan berlepas pada {date}, pukul {time}.', 'Bayaran ansuran bulanan {rm} selama {mo} bulan.',
        'Tarikh akhir pendaftaran ialah {date}.', 'Saya tinggal di tingkat {ord}, unit {unit}.',
        'Tahun {year} merupakan tahun yang mencabar.', 'Kira-kira {pct}% penduduk bersetuju.',
        'Gaji minimum sekarang {rm} sebulan.', 'Kelajuan maksimum ialah {kmh} km/j.']


def numbers(n):
    out = set()
    while len(out) < n * 3:
        t = RNG.choice(NUMT)
        out.add(t.format(rm=rm(), date=date(), time=time_(), phone=phone(),
                         otp=''.join(str(RNG.randint(0, 9)) for _ in range(6)), pct=RNG.randint(1, 90),
                         dec=f'{RNG.randint(1, 9)}.{RNG.randint(1, 9)}', temp=RNG.randint(18, 38),
                         km=RNG.randint(2, 400), kg=RNG.randint(1, 60), h1=time_(), h2=time_(),
                         big=RNG.choice([RNG.randint(100, 999), RNG.randint(1000, 99999)]), ord=RNG.randint(2, 30),
                         mo=RNG.choice([12, 24, 36, 48, 60]), unit=RNG.randint(1, 40), year=RNG.randint(1957, 2030),
                         kmh=RNG.choice([60, 80, 90, 110])))
    return list(out)


# ---------------------------------------------------------------- code-switch
CS = '''Boleh {g} confirm appointment pada hari {h}?
Saya nak cancel booking untuk malam {h}.
Sila check email untuk link pendaftaran.
Delivery {g} akan sampai sebelum pukul {t}.
Payment {g} dah berjaya, terima kasih.
Kalau internet slow, cuba restart router dulu.
Saya dah update app tapi masih ada problem.
Customer service kami boleh dihubungi melalui WhatsApp.
Tolong send location kepada saya sekarang.
Boleh saya tahu status order {g}?
Akaun {g} akan expired minggu depan.
Manager saya nak report itu sebelum hari {h}.
Jangan lupa download resit selepas transfer.
Saya nak claim refund untuk barang rosak.
Voucher discount ini sah sampai hujung bulan.
Sila register dahulu sebelum login.
Team kami akan follow up dengan {g} esok.
Saya perlukan support untuk laptop saya.
Mesej reminder akan dihantar sehari sebelum appointment.
Tolong share video tu dalam group.
Website kami ada sale besar minggu ini.
Okay, saya akan confirm dengan bos saya dulu.
Meeting kita ditunda ke hari {h}, sorry ya.
Password {g} perlu ditukar setiap tiga bulan.
Saya call {g} semula selepas lunch.
Boleh {g} scan dokumen dan email kepada saya?
Free delivery untuk order atas seratus ringgit.
Cashback akan masuk dalam akaun {g} minggu depan.
Kami ada promotion khas untuk member baru.
Saya tengah busy sekarang, nanti saya call balik.
Feedback {g} sangat penting untuk kami.
Sila update alamat terkini dalam profile {g}.
Saya dah book slot untuk servis kereta.
Online banking {g} dah aktif sekarang.
Tolong reset password saya, saya terlupa.
Sorry, line tak clear, boleh ulang?
Barang tu out of stock buat masa ini.
Check in hotel bermula pukul tiga petang.
Saya akan send invoice selepas meeting ni.
Thank you {g}, nanti kami update lagi.
Saya nak tanya pasal bill bulan lepas.
Boleh {g} WhatsApp gambar resit tu?
Parcel {g} dah sampai di office kami.
Nanti saya email quotation kepada {g}.
Sistem tengah down, boleh cuba lagi nanti?
Saya dah transfer, tolong check ya.
Order {g} dah ready untuk pickup.
Boleh {g} share location rumah?
Kami ada promo buy one free one hari ini.
Kad {g} dah expired, perlu renew.
Saya tak dapat login, password salah.
Event tu dibatalkan sebab hujan.
Booking {g} untuk hari {h} dah confirmed.
Boleh {g} reply mesej ni bila free?
Driver dah on the way ke {tp}.
Tolong save nombor ni dalam phone {g}.
Saya akan forward email tu kepada team.
Ada discount untuk student tak?
Link meeting dah dihantar dalam group.
Barang ni ada warranty setahun.
Kalau ada apa-apa issue, terus call saya.
Sorry lambat reply, saya baru habis meeting.
Payment boleh buat secara online atau cash.
Kami terima credit card dan debit card.
Deadline submission ialah hari {h}.
Boss saya nak jumpa {g} untuk interview.
Rider kami tak jumpa rumah {g}.
Saya dah upload semua dokumen ke portal.
Tolong double check maklumat sebelum submit.
Jom lunch sama-sama hari {h}.
Free trial {g} akan tamat minggu ini.
Kami boleh arrange delivery pada hari {h}.
Line internet rumah saya selalu putus.
Saya nak upgrade plan kepada yang lebih laju.
Boleh saya dapatkan refund penuh?
Event kami bermula pukul {t} di {tp}.
Cuba clear cache dan login semula.
Sila follow page kami untuk info terkini.
Nombor tracking dah dihantar melalui SMS.
Saya nak tukar appointment ke hari {h}.'''.splitlines()


def codeswitch(n):
    return list({fill(RNG.choice(CS)) for _ in range(n * 6)})


# ---------------------------------------------------------------- questions / short
QS = '''Apa khabar {g} hari ini?
Bila {g} boleh datang ke pejabat kami?
Di mana {g} tinggal sekarang?
Berapa harga tiket ke {tp}?
Kenapa {g} tidak datang semalam?
Siapa yang akan menguruskan perkara ini?
Bagaimana keadaan {g} sekarang?
Adakah {g} sudah makan?
Boleh saya tahu nama {g}?
Pukul berapa kedai ini tutup?
Wah, cantiknya pemandangan di {tp}!
Alhamdulillah, semuanya berjalan lancar!
Aduh, saya terlupa bawa dompet!
Betul ke {g} nak pindah ke {tp}?
Hebatnya {g}, tahniah!
Mana satu yang {g} lebih suka?
Berapa lama perjalanan ke {tp}?
Sudahkah {g} terima mesej saya?
Seronoknya cuti sekolah tahun ini!
Bolehkah {g} tolong saya sebentar?
Pukul berapa {g} boleh dihubungi?
Boleh {g} ulang nombor telefon tadi?
Berapa ramai yang akan hadir nanti?
Adakah {g} perlukan bantuan tambahan?
Apa nama syarikat {g}?
Kenapa bayaran saya belum masuk?
Bila barang saya akan sampai?
Di mana cawangan yang paling dekat dengan {tp}?
Berapa lama masa menunggu sekarang?
Adakah tempat letak kereta disediakan?
Boleh saya bayar secara ansuran?
Siapa yang patut saya hubungi?
Betulkah harga ini sudah termasuk cukai?
Bagaimana saya boleh menukar alamat?
Ya Allah, banyaknya kerja hari ini!
Syukurlah {g} selamat sampai!
Alamak, hujan lebat pula!
Cepatnya masa berlalu!
Sedapnya makanan di {tp}!
Apa yang berlaku semalam?'''.splitlines()
SHORT = ['Ya, baik.', 'Terima kasih, puan.', 'Terima kasih, encik.', 'Baiklah.', 'Boleh.', 'Tak apa.', 'Sama-sama.',
         'Betul tu.', 'Saya faham.', 'Tunggu sekejap.', 'Maaf ya.', 'Tidak, terima kasih.', 'Ya, betul.',
         'Okey, baik.', 'Sebentar ya.', 'Ada apa-apa lagi?', 'Itu sahaja.', 'Jumpa lagi.', 'Selamat malam.',
         'Selamat pagi.', 'Hello, siapa ni?', 'Baik, faham.', 'Insya-Allah.', 'Boleh saja.', 'Tak boleh.',
         'Nanti dulu.', 'Ya, saya.', 'Silakan.', 'Tak mengapa.', 'Bagus.']


def questions(n):
    return list({fill(RNG.choice(QS)) for _ in range(n * 6)})


def shorts(n):
    out = set()
    while len(out) < n * 2:
        a, b = RNG.sample(SHORT, 2)
        out.add(f'{a} {b}')
    return list(out)


# ---------------------------------------------------------------- selection
def phonemes(norm):
    return G(norm)


def pairs(ph):
    p = ph.replace('ˈ', '').replace('ˌ', '')
    return [p[i:i + 2] for i in range(len(p) - 1) if ' ' not in p[i:i + 2]]


def greedy(pool, k, counts):
    """Pick k sentences, favouring unseen phoneme pairs (lazy greedy over a shuffled pool)."""
    RNG.shuffle(pool)
    chosen = []
    cand = [(s, collections.Counter(pairs(s['phonemes']))) for s in pool]
    while cand and len(chosen) < k:
        best_i, best = 0, -1.0
        for i in range(min(len(cand), 400)):
            sc = sum(c / (1 + counts[p]) for p, c in cand[i][1].items()) / (1 + len(cand[i][1]) ** 0.5)
            if sc > best:
                best_i, best = i, sc
        s, pc = cand.pop(best_i)
        chosen.append(s)
        counts.update(pc)
    return chosen, [c[0] for c in cand]


def row(text, cat):
    norm = normalize(text)
    ph = phonemes(norm)
    if not ph or len(ph) > 400:
        return None
    return dict(category=cat, text=text, norm=norm, teacher=respell(norm, RESPELL), phonemes=ph)


def main():
    pools = {
        'general': tatoeba() + wikipedia(),
        'conversational': conversational(int(TARGET * SHARE['conversational'])),
        'numbers': numbers(int(TARGET * SHARE['numbers'])),
        'codeswitch': codeswitch(int(TARGET * SHARE['codeswitch'])),
        'question': questions(int(TARGET * SHARE['question'])),
        'short': shorts(int(TARGET * SHARE['short'])),
    }
    seen = set()
    rows = {}
    for cat, texts in pools.items():
        rs = []
        for t in texts:
            key = re.sub(r'\W', '', t.lower())
            if key in seen:
                continue
            seen.add(key)
            r = row(t, cat)
            if r:
                rs.append(r)
        rows[cat] = rs
        print(f'{cat:15s} pool {len(rs)}')

    counts = collections.Counter()
    train, test, val, rest = [], [], [], []
    extra_general = [0]
    for cat, share in sorted(SHARE.items(), key=lambda kv: kv[0] == 'general'):
        n = int(TARGET * share) + (extra_general[0] if cat == 'general' else 0)
        pool = rows[cat]
        RNG.shuffle(pool)
        nt, nv = max(4, int(200 * share)), max(2, int(100 * share))
        test += pool[:nt]
        val += pool[nt:nt + nv]
        chosen, left = greedy(pool[nt + nv:], n, counts)
        train += chosen
        short_by = n - len(chosen)
        if short_by > 0:
            print(f'{cat}: short by {short_by}, topping up from general')
            extra_general[0] += short_by
        rest += left
    RNG.shuffle(rest)
    ood = [r for r in rest if r['category'] in ('general', 'conversational', 'question')][:3000]

    out = os.path.join(ROOT, 'data', 'text')
    cols = ['id', 'category', 'text', 'norm', 'teacher', 'phonemes']
    for name, rs, pre in (('corpus_train.tsv', train, 'tr'), ('test_200.tsv', test, 'te'), ('val_100.tsv', val, 'va')):
        with open(os.path.join(out, name), 'w', encoding='utf-8', newline='\n') as f:
            f.write('\t'.join(cols) + '\n')
            for i, r in enumerate(rs):
                r['id'] = f'{pre}{i:05d}'
                f.write('\t'.join(str(r[c]) for c in cols) + '\n')
    with open(os.path.join(out, 'OOD_texts.txt'), 'w', encoding='utf-8', newline='\n') as f:
        for r in ood:
            f.write(r['phonemes'] + '\n')

    allp = collections.Counter(ch for r in train for ch in r['phonemes'] if ch not in ' ˈˌ')
    words = sum(len(r['norm'].split()) for r in train)
    print('train', len(train), 'test', len(test), 'val', len(val), 'ood', len(ood))
    print('by category', collections.Counter(r['category'] for r in train))
    print('est hours at 2.6 words/s:', round(words / 2.6 / 3600, 2))
    print('rarest phonemes', allp.most_common()[-12:])
    pc = collections.Counter(p for r in train for p in pairs(r['phonemes']))
    top = [c for _, c in pc.most_common(400)]
    print('top-400 pairs: min count', min(top), '| pairs under 20:', sum(1 for c in top if c < 20))


if __name__ == '__main__':
    main()
