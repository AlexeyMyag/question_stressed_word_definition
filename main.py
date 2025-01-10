import time

import ffmpeg
import librosa
import numpy as np
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

BUY_RES = [{'results': [{'text': 'ты покупаешь эту квартиру', 'normalized_text': 'Ты покупаешь эту квартиру?', 'start': '0.680s', 'end': '2.360s', 'word_alignments': [{'word': 'ты', 'start': '0.680s', 'end': '0.760s'}, {'word': 'покупаешь', 'start': '0.920s', 'end': '1.440s'}, {'word': 'эту', 'start': '1.640s', 'end': '1.800s'}, {'word': 'квартиру', 'start': '1.920s', 'end': '2.360s'}]}], 'eou': True, 'emotions_result': {'positive': 9.406346e-05, 'neutral': 0.99808097, 'negative': 0.0018248867}, 'processed_audio_start': '0s', 'processed_audio_end': '2.460s', 'backend_info': {'model_name': 'transcribation_hq', 'model_version': 'M-03.002.00-transcribation_hq-01', 'server_version': '03.001.01-rh8-trt10-cuda12-01'}, 'channel': 0, 'speaker_info': {'speaker_id': -1, 'main_speaker_confidence': 1}, 'eou_reason': 'ORGANIC', 'insight': '', 'person_identity': {'age': 'AGE_NONE', 'gender': 'GENDER_NONE', 'age_score': 0, 'gender_score': 0}}]
THIS_RES = [{'results': [{'text': 'ты покупаешь эту квартиру', 'normalized_text': 'Ты покупаешь эту квартиру?', 'start': '0.400s', 'end': '2.040s', 'word_alignments': [{'word': 'ты', 'start': '0.400s', 'end': '0.480s'}, {'word': 'покупаешь', 'start': '0.600s', 'end': '1.120s'}, {'word': 'эту', 'start': '1.280s', 'end': '1.440s'}, {'word': 'квартиру', 'start': '1.600s', 'end': '2.040s'}]}], 'eou': True, 'emotions_result': {'positive': 1.7989063e-05, 'neutral': 0.99938345, 'negative': 0.0005985479}, 'processed_audio_start': '0s', 'processed_audio_end': '2.280s', 'backend_info': {'model_name': 'transcribation_hq', 'model_version': 'M-03.002.00-transcribation_hq-01', 'server_version': '03.001.01-rh8-trt10-cuda12-01'}, 'channel': 0, 'speaker_info': {'speaker_id': -1, 'main_speaker_confidence': 1}, 'eou_reason': 'ORGANIC', 'insight': '', 'person_identity': {'age': 'AGE_NONE', 'gender': 'GENDER_NONE', 'age_score': 0, 'gender_score': 0}}]

def get_graph(file: str, recogn_res):
    print("FILE", file)
    if "http" in file:
        ffmpeg.input(file).output(
            "out.wav",
            format="wav",
            acodec="pcm_s16le",
            ar="48000",
            loglevel="quiet",
            ac=1,
        ).overwrite_output().run()

        file = "out.wav"

    y, sr = librosa.load(file, sr=None)

    # Мел-спектрограмма
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Логарифмическое преобразование для более наглядного отображения
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Интенсивность (энергия сигнала)
    energy = np.sum(S, axis=0)

    # Частотные характеристики
    freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=8000)

    # Показ спектрограммы
    # plt.figure(figsize=(10, 6))
    # librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-frequency spectrogram')
    # plt.show()

    # Длительность сигнала
    duration = librosa.get_duration(y=y, sr=sr)

    # # Форма спектра - спектр мощности
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    #
    # # Показ спектра
    # plt.figure(figsize=(10, 6))
    # librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Spectral Power')
    # plt.show()

    # Вывод информации
    print(f"Duration of the audio: {duration} seconds")

    params = []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for i in recogn_res[0]['results'][0]['word_alignments']:
        # print(i)
        gl_start = float(recogn_res[0]['results'][0]['start'].replace("s", ""))
        gl_end = float(recogn_res[0]['results'][0]['end'].replace("s", ""))

        start = float(i["start"].replace("s", "")) - gl_start
        end = float(i["end"].replace("s", "")) - gl_start

        word_audio = y[int(start * sr):int(end * sr)]

        # Мел-спектрограмма
        S = librosa.feature.melspectrogram(y=word_audio, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Интенсивность (энергия сигнала)
        energy = np.sum(S, axis=0)

        # Частотные характеристики (мел-частоты)
        # freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=8000)
        peak_freqs = librosa.core.hz_to_mel(librosa.core.fft_frequencies(sr=sr))  # Получаем частоты
        peak_freq = peak_freqs[np.argmax(np.mean(S, axis=1))]  # находим максимальную частоту для каждого фрагмента

        # Длительность (время на которое длится слово)
        duration = librosa.get_duration(y=word_audio, sr=sr)

        # Спектр мощности (используем STFT)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(word_audio)), ref=np.max)

        # Добавляем параметры в список
        score = abs(np.mean(D)) + np.mean(energy)

        time_energy = np.linspace(start, end, len(energy))

        # Время для оси X спектра мощности
        time_spectral = np.linspace(start, end, len(D))

        # График энергии
        ax1.plot(time_energy, energy, label=f"Word from {start}s to {end}s")

        # График спектра мощности
        ax2.plot(time_spectral, D, label=f"Word from {start}s to {end}s")

        p = {
            'word': i["word"],
            'start_time': start,
            'end_time': end,
            'energy': np.mean(energy),
            'duration': duration,
            'duration_by_symbol': duration / len(i["word"]),
            'freqs': librosa.mel_to_hz(peak_freq),  # отображаем только первые 10 частот для компактности
            'spectral_power': np.mean(D),
            'score': score
        }

        params.append(p)

        for u in p.keys():
            print(f'{u}: {p[u]}')

        # print("WORD", i["word"], "SCORE", score)

        print("\n")

    # ax1.set_title('Energy of Each Word')
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Energy')
    # # ax1.legend()
    #
    # ax2.set_title('Spectral Power of Each Word')
    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('Spectral Power (dB)')
    # # ax2.legend()
    #
    # plt.tight_layout()
    # plt.show()

    print("\n")
    for param in params:
        print(f"{param['word']}: {param['spectral_power'] / sum([p['spectral_power'] for p in params])}, {param['freqs'] / sum([p['freqs'] for p in params])}, {param['energy'] / sum([p['energy'] for p in params])}. SCORE: {param['score'] / sum([p['score'] for p in params])} {'Q_WORD' if param['score'] / sum([p['score'] for p in params]) > sum([p['score'] for p in params]) / len(params) else ''} A: {sum([p['score'] for p in params])} / {len(params)}")


if __name__ == '__main__':
    # get_graph("record_out_buy.wav", BUY_RES)
    # print("------------")
    # time.sleep(1)
    # get_graph("record_out_this.wav", THIS_RES)
    for u in tqdm(range(1)):
        RES = [{"results": [{"end": "520.727035904s", "text": "хорошо подскажите пожалуйста опция корпоративного такси вам понадобится", "start": "516.687069184s", "normalized_text": "Хорошо, подскажите, пожалуйста, опция корпоративного такси вам понадобится?", "word_alignments": [{"end": "517.007081472s", "word": "хорошо", "start": "516.687069184s"}, {"end": "517.647106048s", "word": "подскажите", "start": "517.127077888s"}, {"end": "518.167068672s", "word": "пожалуйста", "start": "517.727059968s"}, {"end": "518.527090688s", "word": "опция", "start": "518.247055360s"}, {"end": "519.367065600s", "word": "корпоративного", "start": "518.607044608s"}, {"end": "519.687077888s", "word": "такси", "start": "519.447085056s"}, {"end": "519.967113216s", "word": "вам", "start": "519.847051264s"}, {"end": "520.727035904s", "word": "понадобится", "start": "520.127086592s"}]}], "eou_reason": "ORGANIC", "backend_info": {"model_name": "transcribation_hq", "model_version": "M-03.002.00-transcribation_hq-01", "server_version": "03.001.01-rh8-trt10-cuda12-01"}, "speaker_info": {"speaker_id": -1, "main_speaker_confidence": 1}, "emotions_result": {"neutral": 0.9844093, "negative": 0.0042125434, "positive": 0.011378156}, "person_identity": {"age": "AGE_NONE", "gender": "GENDER_NONE", "age_score": 0, "gender_score": 0}, "processed_audio_end": "520.916369408s", "processed_audio_start": "510.847090688s"}]
        file = "https://uhome-minio.k8s.caltat.net/call-records/operator_records/segments/d761dbff-9f03-4f66-b43c-951b1294f41bsegment.wav"
        get_graph(file, RES)

        time.sleep(1)