package com.example.backendjava;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@RestController
public class GenerationController {

    @GetMapping("/")
    public Mono<String> home() {
        return Mono.just("Backend is running");
    }
    private final WebClient webClient;

    public GenerationController(WebClient.Builder builder, @Value("${ml.service.url}") String mlServiceUrl) {
        this.webClient = builder.baseUrl(mlServiceUrl).build();
    }

    @PostMapping("/api/generate")
    public Mono<String> generate(@RequestBody PromptRequest prompt) {
        return this.webClient.post().uri("/generate").bodyValue(prompt)
                .retrieve().bodyToMono(String.class);
    }
}
