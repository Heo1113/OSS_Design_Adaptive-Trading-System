package ats.domain;

/** 주문 체결 상태 (Design 4.2 State machine과 일치) */
public enum OrderStatus { NEW, SUBMITTED, PARTIALLY_FILLED, FILLED, CANCELED }
