// JavaScript for Navigation Buttons
document.getElementById("resourcesButton").addEventListener("click", () => {
    alert("Redirecting to Resources Section!");
    // Code to redirect to resources section
    window.location.href = "#resources"; // Update this with the actual section ID or page URL
});

document.getElementById("faqsButton").addEventListener("click", () => {
    alert("Redirecting to FAQs Section!");
    // Code to redirect to FAQs section
    window.location.href = "#faqs"; // Update this with the actual section ID or page URL
});

document.getElementById("coverLettersButton").addEventListener("click", () => {
    alert("Redirecting to Cover Letters Section!");
    // Code to redirect to cover letters section
    window.location.href = "#coverLetters"; // Update this with the actual section ID or page URL
});

document.getElementById("resumeTemplatesButton").addEventListener("click", () => {
    alert("Redirecting to Resume Templates Section!");
    // Code to redirect to resume templates section
    window.location.href = "#resumeTemplates"; // Update this with the actual section ID or page URL
});

document.getElementById("resumeExamplesButton").addEventListener("click", () => {
    alert("Redirecting to Resume Examples Section!");
    // Code to redirect to resume examples section
    window.location.href = "#resumeExamples"; // Update this with the actual section ID or page URL
});
document.getElementById("applyTemplateBtn").addEventListener("click", function() {
    toggleSections("applyTemplateSection");
});

document.getElementById("resourcesBtn").addEventListener("click", function() {
    toggleSections("resourcesSection");
});

document.getElementById("templatesBtn").addEventListener("click", function() {
    toggleSections("templatesSection");
});

document.getElementById("coverLettersBtn").addEventListener("click", function() {
    toggleSections("coverLettersSection");
});

// Function to toggle the visibility of the sections
function toggleSections(sectionId) {
    const sections = document.querySelectorAll(".content-section");
    sections.forEach(function(section) {
        section.classList.add("hidden");
    });

    const targetSection = document.getElementById(sectionId);
    targetSection.classList.remove("hidden");
}
// Toggle Review Form visibility
document.getElementById("writeReviewBtn").addEventListener("click", function() {
    document.getElementById("reviewFormSection").classList.remove("hidden");
});

// Submit Review
document.getElementById("submitReviewBtn").addEventListener("click", function() {
    const reviewText = document.getElementById("reviewInput").value.trim();

    if (reviewText !== "") {
        const reviewContainer = document.getElementById("reviewsContainer");
        const newReview = document.createElement("div");
        newReview.classList.add("review");
        newReview.innerHTML = `<p><strong>You:</strong> "${reviewText}"</p>`;

        // Add the new review to the container
        reviewContainer.appendChild(newReview);

        // Clear the review input and hide the review form
        document.getElementById("reviewInput").value = "";
        document.getElementById("reviewFormSection").classList.add("hidden");
    } else {
        alert("Please enter a review.");
    }
});
// Get references to elements
const chatContainer = document.querySelector('.chat-container');
const chatBubble = document.createElement('div');
chatBubble.className = 'chat-bubble';
chatBubble.textContent = "Meow! Catch me if you can!";
document.body.appendChild(chatBubble);

// Find the position of the Resume Upload section
const resumeUpload = document.querySelector('.resume-upload');
const resumeUploadTop = resumeUpload.getBoundingClientRect().top; // Get the top position relative to viewport
const resumeUploadHeight = resumeUpload.offsetHeight; // Get the height of the section

// Set the initial position for the cat just above the Resume Upload Section
chatContainer.style.top = (resumeUploadTop - 150) + 'px'; // Place cat 150px above the section

// Random Movement Function (restricted to above Resume Upload)
function moveCat() {
    // Calculate random position within the space above the Resume Upload Section
    const x = Math.random() * (window.innerWidth - 200); // Ensure the cat stays within the viewport horizontally
    const y = Math.random() * (resumeUploadTop - 200); // Ensures the cat stays above the Resume Upload

    // Move the cat container
    chatContainer.style.transform = `translate(${x}px, ${y}px)`;
}

// Initial Movement
moveCat();

// Set an interval for the cat to move around every 3 seconds
setInterval(moveCat, 3000);

// Event Listener for Interaction
chatContainer.addEventListener('click', () => {
    const responses = [
        "Purr! You found me!",
        "Meow! Let's chat!",
        "I'm a sneaky cat, aren't I?",
        "Gotcha! Try again!"
    ];

    // Show a random response in the chat bubble
    chatBubble.textContent = responses[Math.floor(Math.random() * responses.length)];
    chatBubble.style.opacity = 1;

    // Hide the chat bubble after 3 seconds
    setTimeout(() => {
        chatBubble.style.opacity = 0;
    }, 3000);
});


// Resume upload functionality
const uploadForm = document.getElementById("uploadForm");
const resumeOutput = document.getElementById("resumeOutput");

uploadForm.addEventListener("submit", (event) => {
    event.preventDefault(); // Prevent the default form submission
    const fileInput = document.getElementById("resumeFile");
    const file = fileInput.files[0];

    if (file) {
        // Simulate AI processing (replace this with actual backend logic)
        resumeOutput.innerHTML = `<p>Processing your resume: ${file.name}...</p>`;
        setTimeout(() => {
            resumeOutput.innerHTML = `<p>Your resume has been successfully generated!</p>`;
        }, 2000);
    } else {
        resumeOutput.innerHTML = `<p>Please upload a file.</p>`;
    }
});