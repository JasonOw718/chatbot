var chatbotIcon = document.getElementById("chatbot-icon");
var chatContainer = document.getElementById("chat-container");
var chatBox = document.getElementById("chat-box");
var inputBox = document.getElementById("input-box");
var sendButton = document.getElementById("send-button");
var recordButton = document.getElementById("record-button");
var recordDisplay = document.getElementById("recording-text");
var text_to_speech_button = document.getElementById("record-button");
var sendTooltip = document.getElementById("send-tooltip");
var toggleFullscreenButton = document.getElementById("toggle-fullscreen");
let controller;
let url = "";
var currentUserMessage = "";

var questions = [];
var questionButtonRefs = [];
var is_generating = false;
let audio, audioUrl;
var no = 0;
var speakerList = [];
var isLoading = false;
var contain_image = false;
let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let silenceTimer;
const SILENCE_THRESHOLD = 0.01;
const SILENCE_DURATION = 1000;

var loadingAnimation;

function handleKeyPress(event) {
  console.log(is_generating);
  if (event.key === "Enter" && !is_generating) {
    sendButton.click();
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

chatbotIcon.addEventListener("click", function () {
  chatContainer.classList.toggle("hidden");

  if (!chatContainer.classList.contains("hidden")) {
    chatContainer.style.display = "flex";
    chatContainer.offsetHeight;
  }
});

function toggleFullscreen() {
  chatContainer.classList.remove("hidden");
  chatContainer.style.display = "flex";

  setTimeout(() => {
    chatContainer.classList.toggle("fullscreen");
  }, 10);
}

toggleFullscreenButton.addEventListener("click", toggleFullscreen);

sendButton.addEventListener("click", function () {
  if (is_generating) {
    if (AbortController) {
      controller.abort();
      loadingAnimation.remove();
      sendButton.style.backgroundImage = "url('/static/images/submit.png')";
      is_generating = false;
    }
    return;
  }
  var userMessage = inputBox.value.trim();
  if (userMessage !== "") {
    currentUserMessage = userMessage;
    sendMessage(userMessage);
    sendButton.style.backgroundImage = "url('/static/images/square.png')";
    inputBox.value = "";
    removeQuestionButtons();
  }
});

sendButton.onmouseover = function () {
  if (is_generating) sendTooltip.textContent = "Stop response";
};

sendButton.onmouseout = function () {
  if (!is_generating) sendTooltip.textContent = "Send";
};

recordButton.addEventListener("click", async function () {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    var inputContainer = document.querySelector(".input-container");
    inputContainer.classList.add("hide");

    mediaRecorder.addEventListener("dataavailable", (event) => {
      audioChunks.push(event.data);
    });

    mediaRecorder.addEventListener("stop", () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("audio", audioBlob, "audio.webm");

      fetch("/record", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json()) // Parse the response as JSON
        .then((result) => {
          console.log(result); // This will log the parsed object
          if (result.text) {
            document.getElementById("input-box").value = result.text;
          } else if (result.error) {
            console.error("Error:", result.error);
          }
        })
        .catch((error) => {
          console.error("Error:", error);
          // Optionally, display a generic error message to the user
          // document.getElementById("input-box").value = "An error occurred while processing your audio.";
        })
        .finally(() => {
          stopRecording();
        });

      audioChunks = [];
    });

    setTimeout(() => {
      mediaRecorder.start();
      inputBox.style.display = "none";
      sendButton.style.display = "none";
      recordButton.style.display = "none";
      recordDisplay.textContent = "Listening...";
      recordDisplay.classList.add("listening-animation");
      setTimeout(() => recordDisplay.classList.add("show"), 10);

      inputContainer.classList.remove("hide");

      // Start monitoring audio levels
      monitorAudioLevels();
    }, 300);

    // Set a maximum recording time (e.g., 30 seconds)
    setTimeout(() => {
      if (mediaRecorder.state === "recording") {
        mediaRecorder.stop();
      }
    }, 30000);
  } catch (error) {
    console.error("Error:", error);
    recordDisplay.classList.remove("listening-animation", "show");
  }
});

function monitorAudioLevels() {
  const bufferLength = analyser.fftSize;
  const dataArray = new Uint8Array(bufferLength);

  function checkAudioLevel() {
    if (mediaRecorder.state !== "recording") return;

    analyser.getByteTimeDomainData(dataArray);
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      const amplitude = (dataArray[i] - 128) / 128;
      sum += amplitude * amplitude;
    }
    const rms = Math.sqrt(sum / bufferLength);

    if (rms < SILENCE_THRESHOLD) {
      if (!silenceTimer) {
        silenceTimer = setTimeout(() => {
          mediaRecorder.stop();
        }, SILENCE_DURATION);
      }
    } else {
      if (silenceTimer) {
        clearTimeout(silenceTimer);
        silenceTimer = null;
      }
    }

    requestAnimationFrame(checkAudioLevel);
  }

  checkAudioLevel();
}

function stopRecording() {
  var inputContainer = document.querySelector(".input-container");
  inputContainer.classList.add("hide");
  setTimeout(() => {
    inputBox.style.display = "block";
    sendButton.style.display = "block";
    recordButton.style.display = "block";
    recordDisplay.textContent = "";
    recordDisplay.classList.remove("listening-animation", "show");

    inputContainer.classList.remove("hide");
  }, 300);
}

inputBox.addEventListener("keypress", handleKeyPress);

document.querySelectorAll(".card").forEach((card) => {
  card.addEventListener("click", function () {
    if (!is_generating) {
      sendButton.style.backgroundImage = "url('/static/images/square.png')";
      is_generating = true;
      removeQuestionButtons();
      const message = this.getAttribute("data-message");
      sendMessage(message);
    }
  });
});

function displayQuestions() {
  var questionButtons = document.getElementById("question-buttons-container");
  questionButtons.innerHTML = "";
  questions.forEach((question, index) => {
    var questionButton = document.createElement("button");
    questionButton.textContent = question;
    questionButton.classList.add("question-button");

    questionButton.addEventListener("mouseover", function () {
      this.style.backgroundColor = "rgba(66, 66, 67, 0.1)";
      this.style.transform = "scale(1.05)";
    });
    questionButton.addEventListener("mouseout", function () {
      this.style.backgroundColor = "";
      this.style.transform = "";
    });

    questionButton.addEventListener("click", function () {
      sendButton.style.backgroundImage = "url('/static/images/square.png')";
      currentUserMessage = question;
      sendMessage(question, false, parseInt(index));
      removeQuestionButtons();
    });
    questionButtons.appendChild(questionButton);
    questionButtonRefs.push(questionButton);
    setTimeout(() => questionButton.classList.add("show"), 50 * index);
  });

  // Scroll to the bottom after generating question buttons
  chatBox.scrollTop = chatBox.scrollHeight;
  is_generating = false;
}

function removeQuestionButtons() {
  var questionButtons = document.getElementById("question-buttons-container");
  if (questionButtons) {
    questionButtonRefs.forEach((button, index) => {
      setTimeout(() => {
        button.style.opacity = "0";
        button.style.transform = "scale(0.9)";
      }, 50 * index);
      setTimeout(() => button.remove(), 300 + 50 * index);
    });
    questionButtonRefs = [];
    setTimeout(
      () => questionButtons.remove(),
      300 + 50 * questionButtonRefs.length
    );
  }
}

async function sendMessage(message, bool = true, number = 0) {
  controller = new AbortController();
  is_generating = true;
  var messageContainer = document.createElement("div");
  messageContainer.classList.add("message-container");

  var profileContainer = document.createElement("div");
  profileContainer.classList.add("profile-container");

  var profileElement = document.createElement("div");
  profileElement.classList.add("profile-image", "user-profile");
  profileElement.style.backgroundImage = "url('./static/images/user.png')";

  var profileName = document.createElement("span");
  profileName.classList.add("profile-name");
  profileName.textContent = "You";

  var messageElement = document.createElement("div");
  messageElement.classList.add("message", "user-message");
  messageElement.textContent = message;

  profileContainer.appendChild(profileElement);
  profileContainer.appendChild(profileName);
  messageContainer.appendChild(profileContainer);
  messageContainer.appendChild(messageElement);
  chatBox.appendChild(messageContainer);
  console.log("Generating response");
  setTimeout(() => messageElement.classList.add("show"), 10);

  var messageContainer1 = document.createElement("div");
  messageContainer1.classList.add("message-container");

  var profileContainer1 = document.createElement("div");
  profileContainer1.classList.add("profile-container");

  var profileElement1 = document.createElement("div");
  profileElement1.classList.add("profile-image", "bot-profile");
  profileElement1.style.backgroundImage = "url('./static/images/chatbot.png')";

  var profileName1 = document.createElement("span");
  profileName1.classList.add("profile-name");
  profileName1.textContent = "U-Pustaka";

  var messageElement1 = document.createElement("div");
  messageElement1.classList.add("message", "bot-message");

  loadingAnimation = document.createElement("div");
  loadingAnimation.classList.add("lds-ripple");
  loadingAnimation.innerHTML = "<div></div><div></div>";
  messageElement1.appendChild(loadingAnimation);

  profileContainer1.appendChild(profileElement1);
  profileContainer1.appendChild(profileName1);
  messageContainer1.appendChild(profileContainer1);
  messageContainer1.appendChild(messageElement1);
  chatBox.appendChild(messageContainer1);
  setTimeout(() => messageElement1.classList.add("show"), 10);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const response = await fetch("/process_user_message", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_msg: message,
        bool: bool,
        number: number,
      }),
      signal: controller.signal,
    });

    const data = await response.json();
    contain_image = false;
    if (data.image_data) {
      contain_image = true;
      displayImage(data.image_data, data.message);
    }
    loadingAnimation.remove();
    sendBotMessage(data.message, messageElement1);
  } catch (error) {
    if (error.name === "AbortError") {
      console.log(error.name);
    } else {
      console.error("An error occurred:", error);
    }
  }

  url = `/get_questions?user_msg=${encodeURIComponent(message)}`;
}

function createQuestionButtonsContainer(parentElement) {
  // Remove any existing question buttons container
  var existingContainer = document.getElementById("question-buttons-container");
  if (existingContainer) {
    existingContainer.remove();
  }

  var questionButtonsContainer = document.createElement("div");
  questionButtonsContainer.id = "question-buttons-container";
  questionButtonsContainer.style.marginTop = "10px";
  parentElement.appendChild(questionButtonsContainer);
}

document.querySelectorAll(".tooltip-container").forEach((container) => {
  const button = container.querySelector("button");
  const tooltip = container.querySelector(".tooltip");

  button.addEventListener("mouseover", () => {
    tooltip.style.opacity = "1";
  });

  button.addEventListener("mouseout", () => {
    tooltip.style.opacity = "0";
  });
});

function speakMessage(botMessage, speakerButton, tooltip) {
  fetch("/txt_speech", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ botMessage: botMessage }),
  })
    .then((response) => response.blob())
    .then((blob) => {
      audioUrl = URL.createObjectURL(blob);
      audio = new Audio(audioUrl);
      audio.playbackRate = 1.5;
      speakerButton.style.backgroundImage = "url('./static/images/pause.png')";

      audio.addEventListener("ended", () => {
        console.log("Audio playback complete.");
        speakerButton.id = "Listen";
        tooltip.textContent = "Listen";
        speakerButton.style.backgroundImage =
          "url('./static/images/speaker.png')";
      });
      isLoading = false;
      speakerButton.classList.remove("loader");
      speakerButton.disabled = false;
      speakerButton.style.border = "none";
      audio.play();
    })
    .catch((error) => {
      console.error("There was a problem with the fetch operation:", error);
    });
}

function fetchQuestions() {
  fetch(url)
    .then((response) => response.json())
    .then((data) => {
      questions = data;
      displayQuestions();
    })
    .catch((error) => {
      console.error("Error fetching questions:", error);
    });
}

function displayImage(imageData,botMessage) {
  var imgElement = document.createElement("img");
  imgElement.src = "data:image/jpeg;base64," + imageData;
  imgElement.style.maxWidth = "100%";
  imgElement.style.borderRadius = "10px";

  var messageElement = document.createElement("div");
  messageElement.classList.add("message", "bot-message");
  messageElement.appendChild(imgElement);

  var messageContainer = document.createElement("div");
  messageContainer.classList.add("message-container");
  messageContainer.appendChild(messageElement);

  chatBox.appendChild(messageContainer);
  createQuestionButtonsContainer(messageContainer);

  if (chatContainer.classList.contains("fullscreen")) {
    messageElement.scrollIntoView({ behavior: "smooth", block: "end" });
  } else {
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // Add speaker button after the image
  addSpeakerButton(messageElement,botMessage);
}

function addSpeakerButton(parentElement,botMessage) {
  const buttonContainer = document.createElement("div");
  buttonContainer.style.textAlign = "right";
  buttonContainer.style.opacity = "0";
  buttonContainer.style.transition = "opacity 0.3s ease";

  const speakerButton = document.createElement("button");
  speakerButton.id = "Listen";
  speakerButton.style.border = "none";
  speakerButton.style.backgroundImage = "url('./static/images/speaker.png')";
  speakerButton.style.backgroundSize = "cover";
  speakerButton.style.backgroundPosition = "center";
  speakerButton.style.backgroundRepeat = "no-repeat";
  speakerButton.style.cursor = "pointer";
  speakerButton.style.height = "24px";
  speakerButton.style.width = "24px";
  speakerButton.style.transition = "all 0.2s";
  speakerButton.style.padding = "5px";
  speakerButton.style.borderRadius = "50%";
  speakerButton.style.marginTop = "10px";
  speakerList.push(speakerButton);

  const tooltipContainer = document.createElement("div");
  tooltipContainer.style.position = "relative";
  tooltipContainer.style.display = "inline-block";

  const tooltip = document.createElement("div");
  tooltip.textContent = "Listen";
  tooltip.style.position = "absolute";
  tooltip.style.marginTop = "10px";
  tooltip.style.backgroundColor = "#3c4043";
  tooltip.style.color = "white";
  tooltip.style.padding = "3px 8px";
  tooltip.style.borderRadius = "3px";
  tooltip.style.fontSize = "12px";
  tooltip.style.whiteSpace = "nowrap";
  tooltip.style.top = "100%";
  tooltip.style.left = "50%";
  tooltip.style.transform = "translateX(-50%)";
  tooltip.style.opacity = "0";
  tooltip.style.transition = "opacity 0.2s";
  tooltip.style.pointerEvents = "none";

  tooltipContainer.appendChild(speakerButton);
  tooltipContainer.appendChild(tooltip);

  speakerButton.onmouseover = function () {
    if (speakerButton.style.backgroundImage === "") return;

    if (speakerButton.id === "Pausing") tooltip.textContent = "Resume";
    else if (speakerButton.id === "Speaking") tooltip.textContent = "Pause";
    else tooltip.textContent = "Listen";

    this.style.transform = "scale(1.2)";
    this.style.backgroundColor = "rgba(128, 128, 128, 0.2)";
    tooltip.style.opacity = "1";
  };

  speakerButton.onmouseout = function () {
    this.style.transform = "scale(1)";
    this.style.backgroundColor = "transparent";
    tooltip.style.opacity = "0";
  };

  speakerButton.onclick = () => {
    if (isLoading) return;
    console.log(speakerButton.id);

    if (speakerButton.id === "Listen") {
      speakerList.forEach((button) => {
        button.id = "Listen";
        button.style.backgroundImage = "url('./static/images/speaker.png')";
      });
      if (typeof audio !== "undefined") {
        URL.revokeObjectURL(audioUrl);
        audioUrl = null;
        audio.pause();
        audio.src = "";
        audio = null;
      }

      speakerButton.id = "Speaking";
      isLoading = true;
      tooltip.style.opacity = "0";
      prevButtonId = speakerButton.id;
      speakerButton.classList.add("loader");
      speakerButton.style.removeProperty("border");
      speakerButton.style.backgroundImage = "";
      speakerButton.disabled = true;

      speakMessage(botMessage, speakerButton, tooltip);
    } else if (speakerButton.id === "Pausing") {
      speakerButton.id = "Speaking";
      audio.play();
      speakerButton.style.backgroundImage = "url('./static/images/pause.png')";
      tooltip.textContent = "Pause";
    } else if (speakerButton.id === "Speaking") {
      speakerButton.id = "Pausing";
      audio.pause();
      speakerButton.style.backgroundImage = "url('./static/images/resume.png')";
      tooltip.textContent = "Resume";
    }
  };
  buttonContainer.appendChild(tooltipContainer);
  parentElement.appendChild(buttonContainer);
  setTimeout(() => (buttonContainer.style.opacity = "1"), 10);
}

function sendBotMessage(botMessage, messageElement) {
  let index = 0;

  function addNextLetter() {
    if (index < botMessage.length) {
      if (!is_generating) return;

      const char = botMessage[index];
      if (char === "\n") {
        // Handle newlines by adding a line break
        messageElement.appendChild(document.createElement("br"));
      } else {
        messageElement.appendChild(document.createTextNode(char));
      }

      index++;

      if (chatContainer.classList.contains("fullscreen")) {
        messageElement.scrollIntoView({ behavior: "smooth", block: "end" });
      } else {
        chatBox.scrollTop = chatBox.scrollHeight;
      }

      setTimeout(addNextLetter, 10);
    } else {
      sendButton.style.backgroundImage = "url('/static/images/submit.png')";
      if (!contain_image) {
        addSpeakerButton(messageElement,botMessage);
        createQuestionButtonsContainer(messageElement.parentNode);
      }
      fetchQuestions();
    }
  }

  addNextLetter();
}
